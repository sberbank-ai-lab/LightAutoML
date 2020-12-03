"""
Text features
"""

from typing import Any

import torch
from log_calls import record_history

from .base import FeaturesPipeline
from ..utils import get_columns_by_role
from ...dataset.base import LAMLDataset
from ...text.tokenizer import SimpleEnTokenizer, SimpleRuTokenizer, BaseTokenizer
from ...transformers.base import LAMLTransformer, SequentialTransformer, UnionTransformer, ColumnsSelector
from ...transformers.decomposition import SVDTransformer
from ...transformers.text import AutoNLPWrap
from ...transformers.text import TfidfTextTransformer
from ...transformers.text import TokenizerTransformer, ConcatTextTransformer

_model_name_by_lang = {'ru': 'DeepPavlov/rubert-base-cased-conversational',  # "sberbank-ai/sbert_large_nlu_ru" - sberdevices
                       'en': 'bert-base-cased',
                       'multi': 'bert-base-multilingual-cased'}

_tokenizer_by_lang = {'ru': SimpleRuTokenizer,
                      'en': SimpleEnTokenizer,
                      'multi': BaseTokenizer}


@record_history(enabled=False)
class NLPDataFeatures:
    """
    Class contains basic features transformations for text data

    """
    _lang = {'en', 'ru', 'multi'}

    def __init__(self, **kwargs: Any):
        """
        Set default parameters for nlp pipeline constructor

        Args:
            **kwargs:
        """
        if 'lang' in kwargs:
            assert kwargs['lang'] in self._lang, f'Language must be one of: {self._lang}'

        self.lang = 'en'
        self.is_tokenize_autonlp = False
        self.use_stem = False
        self.verbose = False
        self.bert_model = _model_name_by_lang[self.lang]
        self.random_state = 42
        self.device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.model_name = 'wat' if self.device.type == 'cpu' else 'random_lstm' if 'embedding_model' in kwargs else 'random_lstm_bert'
        self.embedding_model = None
        self.svd = True
        self.n_components = 100
        self.is_concat = True
        self.tfidf_params = None
        self.cache_dir = None
        self.train_fasttext = False
        self.embedding_model = None  # path to fasttext model or model with dict interface
        self.transformer_params = None  # params of random_lstm, bert_embedder, borep or wat
        self.fasttext_params = None  # init fasttext params
        self.fasttext_epochs = 2
        self.stopwords = False

        for k in kwargs:
            if kwargs[k] is not None:
                self.__dict__[k] = kwargs[k]


@record_history(enabled=False)
class TextAutoFeatures(FeaturesPipeline, NLPDataFeatures):

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        # TODO: Transformer params to config
        transformers_list = []
        # process texts
        texts = get_columns_by_role(train, 'Text')
        if len(texts) > 0:
            transforms = [ColumnsSelector(keys=texts)]
            if self.is_concat:
                transforms.append(ConcatTextTransformer())
            if self.is_tokenize_autonlp:
                transforms.append(TokenizerTransformer(
                    tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)))
            transforms.append(
                AutoNLPWrap(model_name=self.model_name, embedding_model=self.embedding_model,
                            cache_dir=self.cache_dir, bert_model=self.bert_model, transformer_params=self.transformer_params,
                            random_state=self.random_state, train_fasttext=self.train_fasttext, device=self.device,
                            multigpu=self.multigpu,
                            fasttext_params=self.fasttext_params, fasttext_epochs=self.fasttext_epochs, verbose=self.verbose))

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


@record_history(enabled=False)
class NLPTFiDFFeatures(FeaturesPipeline, NLPDataFeatures):

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        # TODO: Transformer params to config
        transformers_list = []

        # process texts
        texts = get_columns_by_role(train, 'Text')
        if len(texts) > 0:
            transforms = [
                ColumnsSelector(keys=texts),
                TokenizerTransformer(tokenizer=_tokenizer_by_lang[self.lang](is_stemmer=self.use_stem, stopwords=self.stopwords)),
                TfidfTextTransformer(default_params=self.tfidf_params, subs=None, random_state=42)]
            if self.svd:
                transforms.append(SVDTransformer(n_components=self.n_components))

            text_processing = SequentialTransformer(transforms)
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all


@record_history(enabled=False)
class TextBertFeatures(FeaturesPipeline, NLPDataFeatures):

    def create_pipeline(self, train: LAMLDataset) -> LAMLTransformer:
        # TODO: Transformer params to config
        transformers_list = []

        # process texts
        texts = get_columns_by_role(train, 'Text')
        if len(texts) > 0:
            text_processing = SequentialTransformer([

                ColumnsSelector(keys=texts),
                ConcatTextTransformer(),

            ])
            transformers_list.append(text_processing)

        union_all = UnionTransformer(transformers_list)

        return union_all
