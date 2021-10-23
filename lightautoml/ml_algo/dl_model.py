"""Neural net for tabular datasets."""

from lightautoml.utils.installation import __validate_extra_deps


__validate_extra_deps("nlp")


import gc
import logging
import os
import uuid

from copy import copy
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from optuna import Trial
from torch.optim import lr_scheduler
from transformers import AutoTokenizer
from .nn_models import DenseLightModel, DenseModel, ResNetModel, MLP, LinearLayer, SNN
from .tuning.base import Distribution, SearchSpace

from ..ml_algo.base import TabularDataset
from ..ml_algo.base import TabularMLAlgo
from ..pipelines.features.text_pipeline import _model_name_by_lang
from ..pipelines.utils import get_columns_by_role
from ..text.nn_model import CatEmbedder
from ..text.nn_model import ContEmbedder
from ..text.nn_model import TextBert
from ..text.nn_model import TorchUniversalModel
from ..text.nn_model import UniversalDataset
from ..text.trainer import Trainer
from ..text.utils import collate_dict
from ..text.utils import inv_sigmoid
from ..text.utils import inv_softmax
from ..text.utils import is_shuffle
from ..text.utils import parse_devices
from ..text.utils import seed_everything

from ..ml_algo.torch_based.act_funcs import TS
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

logger = logging.getLogger(__name__)

model_by_name = {'dense_light': DenseLightModel, 'dense': DenseModel, 'resnet': ResNetModel,
                 'mlp': MLP, 'dense_layer': LinearLayer, 'snn': SNN}


class TorchModel(TabularMLAlgo):
    """Neural net for tabular datasets.

    default_params:

        - bs: Batch size.
        - num_workers: Number of threads for multiprocessing.
        - max_length: Max sequence length.
        - opt_params: Dict with optim params.
        - scheduler_params: Dict with scheduler params.
        - is_snap: Use snapshots.
        - snap_params: Dict with SE parameters.
        - init_bias: Init last linear bias by mean target values.
        - n_epochs: Number of training epochs.
        - input_bn: Use 1d batch norm for input data.
        - emb_dropout: Dropout probability.
        - emb_ratio: Ratio for embedding size = (x + 1) // emb_ratio.
        - max_emb_size: Max embedding size.
        - bert_name: Name of HuggingFace transformer model.
        - pooling: Type of pooling strategy for bert model.
        - device: Torch device or str.
        - use_cont: Use numeric data.
        - use_cat: Use category data.
        - use_text: Use text data.
        - lang: Text language.
        - deterministic: CUDNN backend.
        - multigpu: Use Data Parallel.
        - path_to_save: Path to save model checkpoints,
          ``None`` - stay in memory.
        - random_state: Random state to take subsample.
        - verbose_inside: Number of steps between
          verbose inside epoch or ``None``.
        - verbose: Verbose every N epochs.

    freeze_defaults:

        - ``True`` :  params may be rewrited depending on dataset.
        - ``False``:  params may be changed only manually or with tuning.

    timer: :class:`~lightautoml.utils.timer.Timer` instance or ``None``.

    """
    _name: str = 'TorchNN'

    _default_params = {
        'bs': 32,
        'num_workers': 4,
        'max_length': 256,
        'opt': torch.optim.Adam,
        'opt_params': {'lr': 1e-5},
        'scheduler_params': {'patience': 5, 'factor': 0.5, 'min_lr': 1e-6, 'verbose': True},
        'is_snap': False,
        'snap_params': {'k': 1, 'early_stopping': True, 'patience': 1, 'swa': False},
        'init_bias': True,
        'n_epochs': 20,
        'input_bn': False,
        'emb_dropout': 0.1,
        'emb_ratio': 3,
        'max_emb_size': 256,
        'bert_name': None,
        'pooling': 'cls',
        'device': torch.device('cuda:0'),
        'use_cont': True,
        'use_cat': True,
        'use_text': True,
        'lang': 'en',
        'deterministic': True,
        'multigpu': False,
        'random_state': 42,
        'efficient': False,
        'model': None,
        'path_to_save': os.path.join('./models/', 'model'),
        'verbose_inside': None,
        'verbose': 1,

        'num_layers': 1,
        'hidden_size_base': 512,
        'drop_rate_base': 0.2,

        'num_blocks': 1,
        'block_size_base': 4,

        'hid_factor_base': 2,
        'drop_rate_base_1': 0.2,
        'drop_rate_base_2': 0.4,

        'sch': lr_scheduler.ReduceLROnPlateau,
        'use_dropout': True,
        'use_noise': False,
    }

    def _infer_params(self):
        if self.params["path_to_save"] is not None:
            self.path_to_save = os.path.relpath(self.params["path_to_save"])
            if not os.path.exists(self.path_to_save):
                os.makedirs(self.path_to_save)
        else:
            self.path_to_save = None

        params = copy(self.params)
        if params["bert_name"] is None:
            params["bert_name"] = _model_name_by_lang[params["lang"]]

        if self.params.get("loss", False):
            self.custom_loss = True
            params["loss"] = self.params["loss"]
        else:
            self.custom_loss = False
            params["loss"] = self.task.losses["torch"].loss

        params["custom_loss"] = self.custom_loss
        params["metric"] = self.task.losses["torch"].metric_func

        is_text = (len(params["text_features"]) > 0) and (params["use_text"]) and (params["device"].type == "cuda")
        is_cat = (len(params["cat_features"]) > 0) and (params["use_cat"])
        is_cont = (len(params["cont_features"]) > 0) and (params["use_cont"])

        torch_model = params["model"]

        if isinstance(torch_model, str):
            assert torch_model in model_by_name, "Wrong model name"
            torch_model = model_by_name[torch_model]
            self.use_custom_model = True

        if isinstance(torch_model, nn.Module):
            self.use_custom_model = True

        if torch_model is None:
            self.use_custom_model = False
            torch_model = model_by_name['dense_light']

        assert issubclass(torch_model, nn.Module), "Wrong model format, only support torch models"

        model = Trainer(
            net=TorchUniversalModel,
            net_params={
                "loss": params["loss"],
                "task": self.task,
                "n_out": params["n_out"],
                "cont_embedder": ContEmbedder if is_cont else None,
                "cont_params": {
                    "num_dims": params["cont_dim"],
                    "input_bn": params["input_bn"],
                }
                if is_cont
                else None,
                "cat_embedder": CatEmbedder if is_cat else None,
                "cat_params": {
                    "cat_dims": params["cat_dims"],
                    "emb_dropout": params["emb_dropout"],
                    "emb_ratio": params["emb_ratio"],
                    "max_emb_size": params["max_emb_size"],
                    "device": params["device"]
                }
                if is_cat
                else None,
                "text_embedder": TextBert if is_text else None,
                "text_params": {
                    "model_name": params["bert_name"],
                    "pooling": params["pooling"],
                }
                if is_text
                else None,
                "bias": params["bias"],
                "torch_model": torch_model,
                **params
            },
            # opt=params["opt"],
            # opt_params=params["opt_params"],
            # n_epochs=params["n_epochs"],
            # device=params["device"],
            # device_ids=params["device_ids"],
            # is_snap=params["is_snap"],
            # snap_params=params["snap_params"],
            # sch=lr_scheduler.ReduceLROnPlateau,
            # scheduler_params=params["scheduler_params"],
            # verbose=params["verbose"],
            # verbose_inside=params["verbose_inside"],
            # metric=params["metric"],
            apex=False,
            **params
        )

        self.train_params = {
            "dataset": UniversalDataset,
            "bs": params["bs"],
            "num_workers": params["num_workers"],
            "tokenizer": AutoTokenizer.from_pretrained(params["bert_name"], use_fast=False) if is_text else None,
            "max_length": params["max_length"],
        }

        return model

    @staticmethod
    def get_mean_target(target, task_name):
        bias = (
            np.array(target.mean(axis=0)).reshape(1, -1).astype(float)
            if (task_name != "multiclass")
            else np.unique(target, return_counts=True)[1]
        )
        bias = (
            inv_sigmoid(bias)
            if (task_name == "binary") or (task_name == "multilabel")
            else inv_softmax(bias)
            if (task_name == "multiclass")
            else bias
        )

        bias[bias == np.inf] = np.nanmax(bias[bias != np.inf])
        bias[bias == -np.inf] = np.nanmin(bias[bias != -np.inf])
        bias[bias == np.NaN] = np.nanmean(bias[bias != np.NaN])

        return bias

    def init_params_on_input(self, train_valid_iterator) -> dict:
        """Get model parameters depending on dataset parameters.

        Args:
            train_valid_iterator: Classic cv-iterator.

        Returns:
            Parameters of model.

        """

        suggested_params = copy(self.default_params)
        suggested_params["device"], suggested_params["device_ids"] = parse_devices(
            suggested_params["device"], suggested_params["multigpu"]
        )

        task_name = train_valid_iterator.train.task.name
        target = train_valid_iterator.train.target
        suggested_params["n_out"] = 1 if task_name != "multiclass" else np.max(target) + 1

        cat_dims = []
        suggested_params["cat_features"] = get_columns_by_role(train_valid_iterator.train, "Category")

        # Cat_features are needed to be preprocessed with LE, where 0 = not known category
        valid = train_valid_iterator.get_validation_data()
        for cat_feature in suggested_params["cat_features"]:
            num_unique_categories = max(max(train_valid_iterator.train[:, cat_feature].data), max(valid[:, cat_feature].data)) + 1
            cat_dims.append(num_unique_categories)
        suggested_params["cat_dims"] = cat_dims

        suggested_params["cont_features"] = get_columns_by_role(train_valid_iterator.train, "Numeric")
        suggested_params["cont_dim"] = len(suggested_params["cont_features"])

        suggested_params["text_features"] = get_columns_by_role(train_valid_iterator.train, "Text")
        suggested_params["bias"] = self.get_mean_target(target, task_name) if suggested_params["init_bias"] else None

        return suggested_params

    def get_dataloaders_from_dicts(self, data_dict):
        logger.debug(f'number of text features: {len(self.params["text_features"])} ')
        logger.debug(f'number of categorical features: {len(self.params["cat_features"])} ')
        logger.debug(f'number of continuous features: {self.params["cont_dim"]} ')

        datasets = {}
        for stage, value in data_dict.items():
            data = {
                name: value.data[cols].values
                for name, cols in zip(
                    ["text", "cat", "cont"],
                    [
                        self.params["text_features"],
                        self.params["cat_features"],
                        self.params["cont_features"],
                    ],
                )
                if len(cols) > 0
            }

            datasets[stage] = self.train_params["dataset"](
                data=data,
                y=value.target.values if stage != "test" else np.ones(len(value.data)),
                w=value.weights.values if value.weights is not None else np.ones(len(value.data)),
                tokenizer=self.train_params["tokenizer"],
                max_length=self.train_params["max_length"],
                stage=stage,
            )

        dataloaders = {
            stage: torch.utils.data.DataLoader(
                datasets[stage],
                batch_size=self.train_params["bs"],
                shuffle=is_shuffle(stage),
                num_workers=self.train_params["num_workers"],
                collate_fn=collate_dict,
                pin_memory=False,
            )
            for stage, value in data_dict.items()
        }
        return dataloaders

    def fit_predict_single_fold(self, train, valid):
        """Implements training and prediction on single fold.

        Args:
            train: Train Dataset.
            valid: Validation Dataset.

        Returns:
            Tuple (model, predicted_values).

        """
        seed_everything(self.params["random_state"], self.params["deterministic"])
        task_name = train.task.name
        target = train.target
        self.params['bias'] = self.get_mean_target(target, task_name) if self.params['init_bias'] else None
        model = self._infer_params()

        model_path = (
            os.path.join(self.path_to_save, f"{uuid.uuid4()}.pickle") if self.path_to_save is not None else None
        )
        # init datasets
        dataloaders = self.get_dataloaders_from_dicts({"train": train.to_pandas(), "val": valid.to_pandas()})

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

    def predict_single_fold(self, model: any, dataset: TabularDataset) -> np.ndarray:
        """Predict target values for dataset.

        Args:
            model: Neural net object or dict or str.
            dataset: Test dataset.

        Return:
            Predicted target values.

        """

        seed_everything(self.params["random_state"], self.params["deterministic"])
        dataloaders = self.get_dataloaders_from_dicts({"test": dataset.to_pandas()})

        if isinstance(model, (str, dict)):
            model = self._infer_params().load_state(model)

        pred = model.predict(dataloaders["test"], "test")

        model.clean()
        del dataloaders, model
        gc.collect()
        torch.cuda.empty_cache()

        return pred

    def _get_default_search_spaces(
            self, suggested_params: Dict, estimated_n_trials: int
    ) -> Dict:
        """Sample hyperparameters from suggested.

        Args:
            trial: Optuna trial object.
            suggested_params: Dict with parameters.
            estimated_n_trials: Maximum number of hyperparameter estimations.

        Returns:
            dict with sampled hyperparameters.

        """
        op_search_space = {}

        op_search_space["opt_params"] = {
            "lr": SearchSpace(
                Distribution.LOGUNIFORM,
                low=1e-5,
                high=1e-2
            ),
            "weight_decay": SearchSpace(
                Distribution.LOGUNIFORM,
                low=0,
                high=1e-2
            )
        }

        op_search_space["bs"] = SearchSpace(
            Distribution.INTUNIFORM,
            low=64,
            high=1024
        )

        # TODO: Add if to agree with user's model params
        # op_search_space["opt"] = SearchSpace(
        #     Distribution.CHOICE,
        #     [torch.optim.Adam, torch.optim.AdamW]
        # )
        #
        # op_search_space["act_fun"] = SearchSpace(
        #     Distribution.CHOICE,
        #     [nn.ReLU, TS, nn.LeakyReLU, nn.Hardswish]
        # )
        #
        # op_search_space["init_bias"] = SearchSpace(
        #     Distribution.CHOICE,
        #     [True, False]
        # )

        # if not self.use_custom_model:
        #     op_search_space["model"] = SearchSpace(
        #         Distribution.CHOICE,
        #         ["dense_light", "dense", "resnet"]
        #     )
        #
        #     if self.params.get("is_cat", False) and len(self.params["cat_dims"]) > 0:
        #         op_search_space["emb_dropout"] = SearchSpace(
        #             Distribution.UNIFORM,
        #             low=0,
        #             high=0.2
        #         )
        #         op_search_space["emb_ratio"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=2,
        #             high=6
        #         )
        #
        #     if op_search_space["model"] == "dense_light":
        #         op_search_space["num_layers"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=1,
        #             high=8
        #         )
        #
        #         hidden_size = ()
        #         drop_rate = ()
        #         hid_high = 1024
        #
        #         # if op_search_space["num_layers"] > 4:
        #         #     hid_high = 512
        #
        #         for layer in range(op_search_space["num_layers"]):
        #             hidden_name = "hidden_size_" + str(layer)
        #             drop_name = "drop_rate_" + str(layer)
        #
        #             op_search_space[hidden_name] = SearchSpace(
        #                 Distribution.INTUNIFORM,
        #                 low=1,
        #                 high=hid_high
        #             )
        #             op_search_space[drop_name] = SearchSpace(
        #                 Distribution.UNIFORM,
        #                 low=0.0,
        #                 high=0.5
        #             )
        #
        #             hidden_size = hidden_size + (op_search_space[hidden_name],)
        #             drop_rate = drop_rate + (op_search_space[drop_name],)
        #
        #         op_search_space["hidden_size"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [hidden_size]
        #         )
        #         op_search_space["drop_rate"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [drop_rate]
        #         )
        #         op_search_space["noise_std"] = SearchSpace(
        #             Distribution.LOGUNIFORM,
        #             low=0,
        #             high=1e-2
        #         )
        #
        #     elif op_search_space["model"] == "dense":
        #         op_search_space["num_blocks"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=1,
        #             high=8
        #         )
        #
        #         block_config = ()
        #         drop_rate = ()
        #
        #         block_high = 8
        #
        #         # if op_search_space["num_blocks"] > 4:
        #         #     block_high = 4
        #
        #         for block in range(op_search_space["num_blocks"]):
        #             block_name = "block_size_" + str(block)
        #             drop_name = "drop_rate_" + str(block)
        #
        #             op_search_space[block_name] = SearchSpace(
        #                 Distribution.INTUNIFORM,
        #                 low=1,
        #                 high=block_high
        #             )
        #             op_search_space[drop_name] = SearchSpace(
        #                 Distribution.UNIFORM,
        #                 low=0.0,
        #                 high=0.5
        #             )
        #
        #             block_config = block_config + (op_search_space[block_name],)
        #             drop_rate = drop_rate + (op_search_space[drop_name],)
        #
        #         op_search_space["block_config"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [block_config]
        #         )
        #
        #         op_search_space["drop_rate"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [drop_rate]
        #         )
        #
        #         op_search_space["num_init_features"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=1,
        #             high=1024
        #         )
        #
        #         op_search_space["compression"] = SearchSpace(
        #             Distribution.UNIFORM,
        #             low=0.0,
        #             high=0.9
        #         )
        #
        #         gr_high = 64
        #         bn_size = 32
        #
        #         # if op_search_space["num_blocks"] > 4:
        #         #     gr_high = 32
        #         #     bn_size = 16
        #
        #         op_search_space["growth_rate"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=8,
        #             high=gr_high
        #         )
        #
        #         op_search_space["bn_size"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=2,
        #             high=bn_size
        #         )
        #
        #     elif op_search_space["model"] == "resnet":
        #         op_search_space["num_layers"] = SearchSpace(
        #             Distribution.INTUNIFORM,
        #             low=1,
        #             high=16
        #         )
        #
        #         hidden_factor = ()
        #         drop_rate = ()
        #         hid_high = 40
        #
        #         # if op_search_space["num_layers"] > 5:
        #         #     hid_high = 20
        #
        #         for layer in range(op_search_space["num_layers"]):
        #             hidden_name = "hidden_factor_" + str(layer)
        #             drop_name = "drop_rate_" + str(layer)
        #
        #             op_search_space[hidden_name] = SearchSpace(
        #                 Distribution.UNIFORM,
        #                 low=1.0,
        #                 high=hid_high
        #             )
        #             op_search_space[drop_name + "_1"] = SearchSpace(
        #                 Distribution.UNIFORM,
        #                 low=0.0,
        #                 high=0.5
        #             )
        #             op_search_space[drop_name + "_2"] = SearchSpace(
        #                 Distribution.UNIFORM,
        #                 low=0.0,
        #                 high=0.5
        #             )
        #
        #             hidden_factor = hidden_factor + (op_search_space[hidden_name],)
        #             drop_rate = drop_rate + (
        #                 (op_search_space[drop_name + "_1"], op_search_space[drop_name + "_2"]),)
        #
        #         op_search_space["hidden_factor"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [hidden_factor]
        #         )
        #
        #         op_search_space["drop_rate"] = SearchSpace(
        #             Distribution.CHOICE,
        #             [drop_rate]
        #         )
        #
        #         op_search_space["noise_std"] = SearchSpace(
        #             Distribution.LOGUNIFORM,
        #             low=0,
        #             high=1e-2
        #         )

        #         op_search_space["drop_connect_rate"] = SearchSpace(
        #              Distribution.UNIFORM,
        #              low=0.0,
        #              high=0.5
        #          )

        return op_search_space

    def _construct_tune_params(self, params, update=False):
        new_params = {}

        new_params["opt_params"] = params.get("opt_params", dict())
        if "lr" in params:
            new_params["opt_params"]["lr"] = params["lr"]
        if "weight_decay" in params:
            new_params["opt_params"]["weight_decay"] = params["weight_decay"]
        elif params.get("weight_decay_bin", -1) == 0:
            new_params["opt_params"]["weight_decay"] = 0

        new_params["scheduler_params"] = params.get("scheduler_params", dict())
        if params["sch"] == StepLR:
            if "step_size" in params:
                new_params["scheduler_params"]["step_size"] = params["step_size"]
            if "gamma" in params:
                new_params["scheduler_params"]["gamma"] = params["gamma"]

            new_params["scheduler_params"] = {
                "step_size": new_params["scheduler_params"]["step_size"],
                "gamma": new_params["scheduler_params"]["gamma"],
            }

        elif params["sch"] == ReduceLROnPlateau:
            if "patience" in params:
                new_params["scheduler_params"]["patience"] = params["patience"]
            if "factor" in params:
                new_params["scheduler_params"]["factor"] = params["factor"]

            new_params["scheduler_params"] = {
                "patience": new_params["scheduler_params"]["patience"],
                "factor": new_params["scheduler_params"]["factor"],
                'min_lr': 1e-6,
            }

        elif params["sch"] == CosineAnnealingLR:
            if "T_max" in params:
                new_params["scheduler_params"]["T_max"] = params["T_max"]
            if "eta_min" in params:
                new_params["scheduler_params"]["eta_min"] = params["eta_min"]
            elif params.get("eta_min_bin", -1) == 0:
                new_params["scheduler_params"]["eta_min"] = 0

            new_params["scheduler_params"] = {
                "T_max": new_params["scheduler_params"]["T_max"],
                "eta_min": new_params["scheduler_params"]["eta_min"],
            }

        else:
            raise ValueError("Worng sch")

        if self.params["model"] == "dense_light" or self.params["model"] == "mlp":
            hidden_size = ()
            drop_rate = ()

            for layer in range(int(params["num_layers"])):
                hidden_name = "hidden_size_base"
                drop_name = "drop_rate_base"

                hidden_size = hidden_size + (params[hidden_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + (params[drop_name],)

            new_params["hidden_size"] = hidden_size
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        elif self.params["model"] == "dense":
            block_config = ()
            drop_rate = ()

            for layer in range(int(params["num_blocks"])):
                block_name = "block_size_base"
                drop_name = "drop_rate_base"

                block_config = block_config + (params[block_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + (params[drop_name],)

            new_params["block_config"] = block_config
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        elif self.params["model"] == "resnet":
            hidden_factor = ()
            drop_rate = ()

            for layer in range(int(params['num_layers'])):
                hidden_name = "hid_factor_base"
                drop_name = "drop_rate_base"

                hidden_factor = hidden_factor + (params[hidden_name],)
                if self.params["use_dropout"]:
                    drop_rate = drop_rate + ((params[drop_name + '_1'], params[drop_name + '_2']),)

            new_params['hid_factor'] = hidden_factor
            if self.params["use_dropout"]:
                new_params["drop_rate"] = drop_rate

        if update:
            self.params.update({**params, **new_params})

        return new_params
