from copy import deepcopy
from typing import List
from typing import Optional

import numpy as np
import pandas as pd
import sklearn.cluster
import torch
import torch.nn.functional as F

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from transformers import AutoModel
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, sentences):
        self.sentences = sentences

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]


class ClusterEmbedder:
    def __init__(
        self,
        embedder_name: str,
        device: str = "cpu",
        batch_size: int = 1,
        lang: str = "russian",
        n_clusters: int = 400,
        cluster_algo: str = "KMeans",
        cluster_params: Optional[dict] = None,
        stop_words: Optional[List[str]] = None,
        word_fetcher_params: Optional[dict] = None,
    ):
        """
        Some info.
        """
        self.embedder = AutoModel(embedder_name)
        self.device = torch.device(device)
        self.embedder.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(embedder_name)
        self.batch_size = batch_size

        self.stemmer = SnowballStemmer(lang)

        word_fetcher_params = word_fetcher_params or {}
        if stop_words is None:
            stop_words = []
        else:
            stop_words.extend([self.preprocess(word) for word in stop_words])

        self.word_fetcher = CountVectorizer(**{**word_fetcher_params, "stop_words": stop_words})
        self.word_fetcher_ = None

        self.n_clusters = n_clusters
        self.cluster_algo = getattr(sklearn.cluster, cluster_algo)
        cluster_params = cluster_params or {}
        self.cluster_algo = self.cluster_algo(**{**cluster_params, "n_clusters": n_clusters})

        self.emb_table = None
        self.embs = None
        self.clust_table = None
        self.clust_to_word = None

    def preprocess(self, text):
        return self.stemmer.stem(text)

    @property
    def voc(self):
        return self.word_fetcher.vocabulary_

    def fit_embeddings(self, text_data: pd.Series):
        self.word_fetcher.fit(text_data)
        sents = list(self.voc)

        def collate_fn(batch):
            return self.tokenizer(batch, paddemb_tableing=True, truncation=True, max_length=32, return_tensors="pt")

        dset = TextDataset(sents)
        dloader = DataLoader(dset, shuffle=False, batch_size=self.batch_size, collate_fn=collate_fn)
        embs = []
        with torch.no_grad():
            for data in dloader:
                data["input_ids"] = data["input_ids"].to(self.device)
                data["token_type_ids"] = data["token_type_ids"].to(self.device)
                data["attention_mask"] = data["attention_mask"].to(self.device)
                out_embs = self.embedder(**data).pooler_output
                out_embs = F.normalize(out_embs).cpu().detach().numpy()
                embs.append(out_embs)

        self.embs = np.vstack(embs)
        self.emb_table = {word: emb for word, emb in zip(self.voc, self.embs)}

    def get_embeddings(self):
        return self.emb_table

    def get_clust_table(self):
        return self.clust_table

    def get_clust_words(self, cluster):
        return [self.voc[i2v] for i2v in self.clust_to_word[cluster].tolist()]

    def fit_clusters(self):
        assert self.embs is not None, "Embeddings must be fitted"
        clusters = self.cluster_algo.fit_predict(self.embs)

        self.clust_table = {word: clust for word, clust in zip(self.voc, clusters)}
        c2w = [[] for i in range(self.n_clusters)]
        for iw, c in enumerate(self.cluster_table.values()):
            c2w[c].append(iw)
        self.clust_to_word = [np.array(iws) for iws in c2w]
        self.word_fetcher_ = deepcopy(self.word_fetcher)
        self.word_fetcher_.vocabulary_ = self.clust_table

    def transform(self, text_data: pd.Series):
        assert self.word_fetcher_ is not None, "First, fit clusters"
        encoded = self.word_fetcher_.transform(text_data)
        return encoded[:, : self.n_clusters]

    def get_cluster_weights(
        self, text_data: pd.Series, target: np.array, task: str = "binary", scaler: Optional[str] = None, **kwargs
    ):
        assert task in {"binary", "reg"}, "Task must be 'binary' or 'reg'"
        encoded = self.transform(text_data)
        if task == "binary":
            lin_model = LogisticRegression(**kwargs)
        else:
            lin_model = LinearRegression(**kwargs)

        if scaler is not None:
            scaler = getattr(sklearn.preprocessing, scaler)
            scaler = scaler()
            encoded = scaler.fit_transform(encoded)

        lin_model.fit(encoded, target)
        return lin_model.coef_.flatten()
