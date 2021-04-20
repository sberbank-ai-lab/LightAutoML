import re
import itertools


from typing import Iterable, Optional, List, Any

import numpy as np
import pandas as pd

from functools import partial
from sklearn.utils import check_random_state
from sklearn.linear_model import Ridge, lars_path
from sklearn.metrics import pairwise_distances 

from ...pipelines.features.text_pipeline import _tokenizer_by_lang

from collections import defaultdict
from .utils import IndexedString, draw_html


class TextExplanation:
    def __init__(self, index_string: IndexedString,
                 task_name: str, 
                 class_names: Optional[List[Any]] = None,
                 random_state=None):
        
        self.idx_str = index_string
        # todo: regression-mode
        assert task_name in ['binary', 'multiclass']
        self.task_name = task_name
        
        self.class_names = class_names
        self.random_state = check_random_state(random_state)
        self.instance = {}
        
    
    def as_list(self, label=1, **kwargs):
        ans = self.instance[label]['feature_weights']
        ans = [(x[0], float(x[1])) for x in ans]
        return ans
    
    def as_features(self, label=1, add_not_rel=False, normalize=False):
        fw = self.instance[label]['feature_weights']
        norm_const = 1.0
        if normalize:
            norm_const = 1 / abs(fw[0][1])
        fw = dict(fw)
        if add_not_rel:
            fw = dict(self.instance[label]['feature_weights'])
            weights = np.zeros_like(self.idx_str.as_np_,
                                    dtype=np.float32)
            for k, v in fw.items():
                weights[self.idx_str.pos[k]] = v
                
            ans = [(k, float(w) * norm_const) \
                   for k, w in zip(self.idx_str.as_np_, weights)]
        else:
            ans = [(self.idx_str.word(k), float(v) * norm_const)\
                   for k, v in ans.items()]
            
        return ans
    
    def as_map(self, label):
        return {k : v['feature_weights'] \
                for k, v in self.instance.items()}

    def as_html(self, label):
        weight_string = self.as_features(label, True, True)
        return draw_html(weight_string)
        
    def visualize_in_notebook(self, label):
        from IPython.display import HTML, display_html
        
        raw_html = self.as_html(label)
        if display:
            display_html(HTML(raw_html))

            
class LimeTextExplainer:
    
    
    def __init__(self, automl, kernel=None, kernel_width=25,
                 feature_selection='none',
                 force_order=False, model_regressor=None,
                 distance_metric='cosine', random_state=0):
        self.automl = automl
        self.task_name = automl.reader.task.name
        self.roles = automl.reader.roles
        self.kernel_width = kernel_width
        if kernel is None:
            def kernel(d, kernel_width):
                return np.sqrt(np.exp(- d**2 / kernel_width**2))
        assert callable(kernel)
        
        self.kernel_fn = partial(kernel, kernel_width=kernel_width)
        self.random_state = check_random_state(random_state)
        
        if model_regressor is None:
            model_regressor = Ridge(
                alpha=1, fit_intercept=True, 
                random_state=self.random_state)
            
        self.distil_model = model_regressor
        
        # todo: forward selection and higher weights
        # and auto-mode
        
        assert feature_selection in ['none', 'lasso']
        
        self.feature_selection = feature_selection
        self.force_order = force_order
        
        lang = automl.text_params['lang']
        self.tokenizer = _tokenizer_by_lang[lang](is_stemmer=False)
        self.distance_metric = distance_metric
        
        class_names = automl.reader.class_mapping
        if class_names == None:
            class_names = np.arange(automl.reader._n_classes)
        else:
            class_names = list(class_mapping.values())
            
        self.class_names = class_names
        
        
    def explain_instance(self, data: pd.Series, perturb_column: str,
                         labels: Iterable =(1,), n_features: int = 10,
                         n_samples: int = 5000):
        """
        Args:
            data: 
        
        """
        assert self.roles[perturb_column].name == 'Text', \
            'Column is not text column'
        assert n_samples > 1, 'Number of generated samples must be > 0'
        
        
        data, y, dst, expl = self._get_perturb_dataset(
            data, perturb_column, n_samples)
        for label in labels:
            expl.instance[label] = self._explain_dataset(
                data, y, dst, label, n_features)
        
        return expl

    def _get_perturb_dataset(self, data, perturb_column, n_samples):
        text = data[perturb_column]
        idx_str = IndexedString(text, self.tokenizer, self.force_order)
        n_words = idx_str.n_words
        samples = self.random_state.randint(1, n_words + 1, n_samples - 1)
        raw_dataset = [data.copy()]
        dataset = np.ones((n_samples, n_words))
        
        for i, size in enumerate(samples, start=1):
            off_tokens = self.random_state.choice(range(n_words), size)
            data_ = data.copy()
            p_text = idx_str.inverse_removing(off_tokens)
            data_[perturb_column] = p_text
            raw_dataset.append(data_)
            dataset[i, off_tokens] = 0
        
        raw_dataset = pd.DataFrame(raw_dataset)
        
        pred = self.automl.predict(raw_dataset).data
        if self.task_name == 'binary':
            pred = np.concatenate([1 - pred, pred], axis=1)


        distance = pairwise_distances(dataset, dataset[0].reshape(1, -1),
                                      metric=self.distance_metric).ravel()
        
        expl = TextExplanation(idx_str,
                               self.task_name,
                               self.class_names,
                               self.random_state)
        
        return dataset, pred, distance * 100, expl
        
    def _explain_dataset(self, data, y, dst, label, n_features):
        weights = self.kernel_fn(dst)
        y = y[:, label]
        features = self._feature_selection(
            data, y, weights, n_features,
            mode=self.feature_selection)
        model = self.distil_model
        model.fit(data[:, features], y, sample_weight=weights)
        score = model.score(data[:, features], y,
                            sample_weight=weights)
        
        pred = model.predict(data[0, features].reshape(1, -1))
        feature_weights = list(sorted(zip(features, model.coef_),
                                      key=lambda x: np.abs(x[1]),
                                      reverse=True))
        res = {
            'bias': model.intercept_,
            'feature_weights': feature_weights,
            'score': score,
            'pred': pred
        }
        
        return res
                
        
    def _feature_selection(self, data, y, weights, n_features, mode='none'):
        if mode == 'none':
            return np.arange(data.shape[1])
        if mode == 'lasso':
            weighted_data = ((data - np.average(data, axis=0, weights=weights))
                             * np.sqrt(weights[:, np.newaxis]))
            weighted_y = ((y - np.average(y, weights=weights))
                               * np.sqrt(weights))
            
            features = np.arange(weighted_data.shape[1])
            _, _, coefs = lars_path(weighted_data,
                                    weighted_y,
                                    method='lasso',
                                    verbose=False)
            
            for i in range(len(coefs.T) - 1, 0, -1):
                features = coefs.T[i].nonzero()[0]
                if len(features) <= n_features:
                    break
            
            return features
        