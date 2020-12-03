"""
Deep learning models
"""

import gc
import os
import uuid
from copy import copy

import numpy as np
import torch
from log_calls import record_history
from torch.optim import lr_scheduler
from torch.utils.data import Dataset

from transformers import AutoTokenizer
from ..ml_algo.base import TabularMLAlgo
from ..pipelines.features.text_pipeline import _model_name_by_lang
from ..pipelines.utils import get_columns_by_role
from ..text.nn_model import TorchUniversalModel, ContEmbedder, CatEmbedder, TextBert, UniversalDataset
from ..text.trainer import Trainer
from ..text.utils import seed_everything, parse_devices, collate_dict, is_shuffle, inv_softmax, inv_sigmoid


@record_history(enabled=False)
class TorchModel(TabularMLAlgo):
    """
        TorchNN.
    """
    _name: str = 'TorchNN'
    _default_params = {'random_state': 42,
                       'bs': 16,
                       'num_workers': 4,
                       'max_length': 256,
                       'path_to_save': os.path.join('./models/', 'model'),
                       'opt_params': {'lr': 1e-4, },
                       'scheduler_params': {'patience': 5, 'factor': 0.5, 'verbose': True},
                       'is_snap': False,
                       'snap_params': {'k': 1, 'early_stopping': True, 'patience': 1, 'swa': False},
                       'init_bias': True,
                       'verbose_inside': None,
                       'verbose': 1,
                       'n_epochs': 20,
                       'input_bn': False,
                       'emb_dropout': 0.1,
                       'emb_ratio': 3,
                       'max_emb_size': 50,
                       'bert_name': None,
                       'pooling': 'cls',
                       'device': [0],  # or '0' or 'cuda:0' / cpu or torch.device("cuda:0")
                       'use_cont': True,
                       'use_cat': True,
                       'use_text': True,
                       'lang': 'en',
                       'deterministic': True,
                       'multigpu': False
                       }

    def _infer_params(self):
        if self.params['path_to_save'] is not None:
            self.path_to_save = os.path.relpath(self.params['path_to_save'])
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
        else:
            self.path_to_save = None

        params = copy(self.params)
        if params['bert_name'] is None:
            params['bert_name'] = _model_name_by_lang[params['lang']]

        params['loss'] = self.task.losses['torch'].loss
        params['metric'] = self.task.losses['torch'].metric_func

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params['device'].type == 'cuda')
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        model = Trainer(net=TorchUniversalModel,
                        net_params={'loss': params['loss'],
                                    'task': self.task,
                                    'n_out': params['n_out'],
                                    'cont_embedder': ContEmbedder if is_cont else None,
                                    'cont_params': {'num_dims': params['cont_dim'],
                                                    'input_bn': params['input_bn']} if is_cont else None,
                                    'cat_embedder': CatEmbedder if is_cat else None,
                                    'cat_params': {'cat_dims': params['cat_dims'], 'emb_dropout': params['emb_dropout'],
                                                   'emb_ratio': params['emb_ratio'],
                                                   'max_emb_size': params['max_emb_size']} if is_cat else None,
                                    'text_embedder': TextBert if is_text else None,
                                    'text_params': {'model_name': params['bert_name'],
                                                    'pooling': params['pooling']} if is_text else None,
                                    'bias': params['bias'],
                                    },
                        opt=torch.optim.Adam,
                        opt_params=params['opt_params'],
                        n_epochs=params['n_epochs'],
                        device=params['device'],
                        device_ids=params['device_ids'],
                        is_snap=params['is_snap'],
                        snap_params=params['snap_params'],
                        sch=lr_scheduler.ReduceLROnPlateau,
                        scheduler_params=params['scheduler_params'],
                        verbose=params['verbose'],
                        verbose_inside=params['verbose_inside'],
                        metric=params['metric'],
                        apex=False,
                        )

        self.train_params = {'dataset': UniversalDataset, 'bs': params['bs'], 'num_workers': params['num_workers'],
                             'tokenizer': AutoTokenizer.from_pretrained(params['bert_name']) if is_text else None,
                             'max_length': params['max_length']}

        return model

    @staticmethod
    def get_mean_target(target, task_name):
        bias = np.array(target.mean(axis=0)).reshape(1, -1).astype(float) if (task_name != 'multiclass') else \
            np.unique(target, return_counts=True)[1]
        bias = inv_sigmoid(bias) if (task_name == 'binary') or (task_name == 'multilabel') else inv_softmax(bias) if (
                task_name == 'multiclass') else bias

        bias[bias == np.inf] = np.nanmax(bias[bias != np.inf])
        bias[bias == -np.inf] = np.nanmin(bias[bias != -np.inf])
        bias[bias == np.NaN] = np.nanmean(bias[bias != np.NaN])

        return bias

    def init_params_on_input(self, train_valid_iterator) -> dict:

        suggested_params = copy(self.default_params)
        suggested_params['device'], suggested_params['device_ids'] = parse_devices(suggested_params['device'],
                                                                                   suggested_params['multigpu'])

        task_name = train_valid_iterator.train.task.name
        target = train_valid_iterator.train.target
        suggested_params['n_out'] = 1 if task_name != 'multiclass' else np.max(target) + 1

        cat_dims = []
        suggested_params['cat_features'] = get_columns_by_role(train_valid_iterator.train, 'Category')
        for cat_feature in suggested_params['cat_features']:
            num_unique_categories = max(train_valid_iterator.train[:, cat_feature].data) + 1
            cat_dims.append(num_unique_categories)
        suggested_params['cat_dims'] = cat_dims

        suggested_params['cont_features'] = get_columns_by_role(train_valid_iterator.train, 'Numeric')
        suggested_params['cont_dim'] = len(suggested_params['cont_features'])

        suggested_params['text_features'] = get_columns_by_role(train_valid_iterator.train, 'Text')
        suggested_params['bias'] = self.get_mean_target(target, task_name) if suggested_params['init_bias'] else None

        return suggested_params

    def get_dataloaders_from_dicts(self, data_dict):
        print(f'n text: {len(self.params["text_features"])} ', f'n cat: {len(self.params["cat_features"])} ',
              f'n cont: {self.params["cont_dim"]} ')

        datasets = {stage: self.train_params['dataset'](data={name: value.data[cols].values for name, cols in
                                                           zip(['text', 'cat', 'cont'],
                                                               [self.params["text_features"],
                                                                self.params["cat_features"],
                                                                self.params["cont_features"]]) if len(cols) > 0},
                                                        y=value.target.values if stage != 'test' else np.ones(len(value.data)),
                                                        w=value.weights.values if value.weights is not None else np.ones(
                                                            len(value.data)),
                                                        tokenizer=self.train_params['tokenizer'],
                                                        max_length=self.train_params['max_length'],
                                                        stage=stage) for stage, value in
                    data_dict.items()}

        dataloaders = {stage: torch.utils.data.DataLoader(datasets[stage],
                                                          batch_size=self.train_params['bs'],
                                                          shuffle=is_shuffle(stage),
                                                          num_workers=self.train_params['num_workers'],
                                                          collate_fn=collate_dict,
                                                          pin_memory=False) for stage, value in data_dict.items()}
        return dataloaders

    def fit_predict_single_fold(self, train, valid):
        """

        Args:
            train:
            valid:

        Returns:

        """
        seed_everything(self.params['random_state'], self.params['deterministic'])
        model = self._infer_params()

        model_path = os.path.join(self.path_to_save, f'{uuid.uuid4()}.pickle') if self.path_to_save is not None else None
        # init datasets
        dataloaders = self.get_dataloaders_from_dicts({'train': train, 'val': valid})

        val_pred = model.fit(dataloaders)

        if model_path is None:
            model_path = model.state_dict(model_path)
        else:
            model.state_dict(model_path)

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()
        return model_path, val_pred

    def predict_single_fold(self, model, dataset) -> np.ndarray:
        seed_everything(self.params['random_state'], self.params['deterministic'])
        dataloaders = self.get_dataloaders_from_dicts({'test': dataset})

        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders['test'], 'test')

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred
