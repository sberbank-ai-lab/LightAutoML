"""
DL transformers
"""

import gc

import numpy as np
import torch
import torch.nn as nn
from log_calls import record_history
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Any, Dict, Optional, Sequence

from transformers import AutoModel
from .dp_utils import CustomDataParallel
from .sentence_pooling import SequenceMaxPooler, SequenceAvgPooler, SequenceSumPooler, SequenceClsPooler, \
    SequenceIndentityPooler
from .utils import seed_everything, parse_devices, collate_dict, single_text_hash, _dtypes_mapping

pooling_by_name = {'mean': SequenceAvgPooler,
                   'sum': SequenceSumPooler,
                   'max': SequenceMaxPooler,
                   'cls': SequenceClsPooler,
                   'none': SequenceIndentityPooler}


@record_history(enabled=False)
class DLTransformer(TransformerMixin):
    """Deep Learning based sentence embeddings."""

    model_params = {'embed_size': 300, 'hidden_size': 256, 'pooling': 'mean', 'num_layers': 1}
    loader_params = {'batch_size': 1024, 'shuffle': False, 'num_workers': 4}
    dataset_params = {'embedding_model': None, 'max_length': 200, 'embed_size': 300}
    embedding_model_params = {'model_name': 'bert-base-cased'}

    def __init__(self, model, model_params: Dict, dataset, dataset_params: Dict, loader_params: Dict,
                 device: str = 'cuda', random_state: int = 42,
                 embedding_model: Optional = None, embedding_model_params: Dict[str, Dict] = None,
                 multigpu: bool = False,
                 verbose: bool = False):
        """Class to compute sentence embeddings from words embeddings.

        Args:
            model: torch model for aggregation word embeddings into sentence embedding
            model_params: dict with model parameters
            dataset: torch dataset
            dataset_params: dict with dataset params
            loader_params: dict with params for torch dataloader
            device: string with torch device type or device ids. I.e: '0,2'
            random_state: determines random number generation
            embedding_model: torch word embedding model if dataset do not return embeddings
            embedding_model_params: dict with embedding model params
            multigpu: use data parallel for multiple GPU
            verbose: show tqdm progress bar

        """
        super(DLTransformer, self).__init__()

        self.device, self.device_ids = parse_devices(device, multigpu)

        self.random_state = random_state
        self.verbose = verbose
        seed_everything(random_state)
        self.model_params.update(model_params)
        self.model = model(**self.model_params)

        self.embedding_model = None
        if embedding_model is not None:
            if embedding_model_params is not None:
                self.embedding_model_params.update(embedding_model_params)
            self.embedding_model = embedding_model(**self.embedding_model_params)

        self.dataset = dataset
        self.dataset_params.update(dataset_params)
        self.loader_params.update(loader_params)

    def get_name(self) -> str:
        """Module name.

        Returns:
            string with module name

        """
        if self.embedding_model is None:
            name = self.model.get_name()
        else:
            name = self.model.get_name() + '_' + self.embedding_model.get_name()
        return name

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape

        """
        return self.model.get_out_shape()

    def fit(self, data: Any):
        return self

    @torch.no_grad()
    def transform(self, data: Sequence[str]) -> np.ndarray:

        dataset = self.dataset(data, **self.dataset_params)
        loader = DataLoader(dataset, collate_fn=collate_dict, **self.loader_params)

        result = []
        if self.verbose:
            loader = tqdm(loader)

        self.model = self.model.to(self.device)
        if self.device_ids is not None:
            self.model = CustomDataParallel(self.model, device_ids=self.device_ids)
        self.model.eval()

        if self.embedding_model is not None:
            self.embedding_model.to(self.device)
            if self.device_ids is not None:
                self.embedding_model = CustomDataParallel(self.embedding_model, device_ids=self.device_ids)
            self.embedding_model.eval()

        for sample in loader:
            data = {i: sample[i].long().to(self.device) if _dtypes_mapping[i] == 'long' else sample[i].to(self.device) for i in
                    sample.keys()}
            if self.embedding_model is not None:
                embed = self.embedding_model(data)
                if 'attention_mask' in data:
                    length = torch.sum(data['attention_mask'], dim=1)
                else:
                    length = (torch.ones(len(embed)) * self.dataset_params['max_length']).to(self.device).long()
                data = {'text': embed, 'length': length}
            embed = self.model(data).detach().cpu().numpy()
            result.append(embed.astype(np.float32))

        result = np.vstack(result)
        self.model.to(torch.device('cpu'))
        if isinstance(self.model, CustomDataParallel):
            self.model = self.model.module

        if self.embedding_model is not None:
            self.embedding_model.to(torch.device('cpu'))

            if isinstance(self.embedding_model, CustomDataParallel):
                self.embedding_model = self.embedding_model.module

        del loader, dataset, data
        gc.collect()
        torch.cuda.empty_cache()

        return result


@record_history(enabled=False)
def position_encoding_init(n_pos: int, embed_size: int) -> torch.Tensor:
    """Compute positional embedding matrix.

    Args:
        n_pos: len of sequence
        embed_size: size of output sentence embedding

    Returns:
        torch tensor with all positional embeddings

    """
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / embed_size) for j in range(embed_size)]
        if pos != 0 else np.zeros(embed_size) for pos in range(n_pos)])
    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2])
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2])
    return torch.from_numpy(position_enc).float()


@record_history(enabled=False)
class BOREP(nn.Module):
    """Class to compute Bag of Random Embedding Projections sentence embeddings from words embeddings."""

    name = 'BOREP'
    _poolers = {"max", "mean", "sum"}

    def __init__(self, embed_size: int = 300, proj_size: int = 300, pooling: str = 'mean',
                 max_length: int = 200, init: str = 'orthogonal', pos_encoding: bool = False,
                 **kwargs: Any):
        """Bag of Random Embedding Projections sentence embeddings.

        Args:
            embed_size: size of word embeddings
            proj_size: size of output sentence embedding
            pooling: str with pooling type:
                - max: maximum on seq_len dimension for non masked inputs
                - mean: mean on seq_len dimension for non masked inputs
                - sum: sum on seq_len dimension for non masked inputs
            max_length: maximum length of sentence
            init: type of weight initialization:
                - orthogonal: orthogonal init
                - normal: normal with std 0.1
                - uniform: uniform from -0.1 to 0.1
                - kaiming: uniform kaiming init
                - xavier: uniform xavier init
            pos_encoding: add positional embedding
            kwargs: ignored params

        """
        super(BOREP, self).__init__()
        self.embed_size = embed_size
        self.proj_size = proj_size
        self.pos_encoding = pos_encoding

        if self.pos_encoding:
            self.pos_code = position_encoding_init(max_length, self.embed_size).view(1, max_length, self.embed_size)

        self.pooling = pooling_by_name[pooling]()

        self.proj = nn.Linear(self.embed_size, self.proj_size, bias=False)

        if init == "orthogonal":
            nn.init.orthogonal_(self.proj.weight)
        elif init == "normal":
            nn.init.normal_(self.proj.weight, std=0.1)
        elif init == "uniform":
            nn.init.uniform_(self.proj.weight, a=-0.1, b=0.1)
        elif init == "kaiming":
            nn.init.kaiming_uniform_(self.proj.weight)
        elif init == "xavier":
            nn.init.xavier_uniform_(self.proj.weight)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape

        """
        return self.proj_size

    def get_name(self) -> str:
        """Module name.

        Returns:
            string with module name

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        x = inp['text']
        batch_size, batch_max_length = x.shape[0], x.shape[1]
        if self.pos_encoding:
            x = x + self.pos_code[:, :batch_max_length, :].to(x.device)
        x = x.contiguous().view(batch_size * batch_max_length, -1)
        x = self.proj(x)
        out = x.contiguous().view(batch_size, batch_max_length, -1)
        x_length = (torch.arange(out.shape[1])[None, :].to(out.device) < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)

        return out


@record_history(enabled=False)
class RandomLSTM(nn.Module):
    """Class to compute Random LSTM sentence embeddings from words embeddings."""

    name = 'RandomLSTM'
    _poolers = ("max", "mean", "sum")

    def __init__(self, embed_size: int = 300, hidden_size: int = 256, pooling: str = 'mean',
                 num_layers: int = 1, **kwargs: Any):
        """Random LSTM sentence embeddings.

        Args:
            embed_size: size of word embeddings
            hidden_size: size of hidden dimensions of LSTM
            pooling: str with pooling type:
                - max: maximum on seq_len dimension for non masked inputs
                - mean: mean on seq_len dimension for non masked inputs
                - sum: sum on seq_len dimension for non masked inputs
            num_layers: number of lstm layers
            kwargs: ignored params

        """
        super(RandomLSTM, self).__init__()
        if pooling not in self._poolers:
            raise ValueError("pooling - {} - not in the list of available types {}".format(pooling, self._poolers))

        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers,
                            bidirectional=True, batch_first=True)

        self.pooling = pooling_by_name[pooling]()

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape

        """
        return self.hidden_size * 2

    def get_name(self) -> str:
        """Module name.

        Returns:
            string with module name

        """
        return self.name

    @torch.no_grad()
    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        out, _ = self.lstm(inp['text'])
        x_length = (torch.arange(out.shape[1])[None, :].to(out.device) < inp['length'][:, None])[:, :, None]
        out = self.pooling(out, x_length)
        return out


@record_history(enabled=False)
class BertEmbedder(nn.Module):
    """Class to compute HuggingFace transformers words or sentence embeddings."""

    name = 'BertEmb'
    _poolers = {"cls", "max", "mean", "sum", "none"}

    def __init__(self, model_name: str, pooling: str = 'none', **kwargs: Any):
        """Bert sentence or word embeddings.

        Args:
            model_name: name of transformers model
            pooling: str with pooling type:
                - cls: use CLS token for sentence embedding from last hidden state
                - max: maximum on seq_len dimension for non masked inputs from last hidden state
                - mean: mean on seq_len dimension for non masked inputs from last hidden state
                - sum: sum on seq_len dimension for non masked inputs from last hidden state
                - none: don not use pooling (for RandomLSTM pooling strategy)
            kwargs: ignored params

        """
        super(BertEmbedder, self).__init__()
        if pooling not in self._poolers:
            raise ValueError("pooling - {} - not in the list of available types {}".format(pooling, self._poolers))

        self.pooling = pooling_by_name[pooling]()

        self.model_name = model_name
        self.transformer = AutoModel.from_pretrained(model_name)

    def forward(self, inp: Dict[str, torch.Tensor]) -> torch.Tensor:
        encoded_layers, _ = self.transformer(input_ids=inp['input_ids'],
                                             attention_mask=inp['attention_mask'],
                                             token_type_ids=inp['token_type_ids'], return_dict=False)

        encoded_layers = self.pooling(encoded_layers, inp['attention_mask'])

        return encoded_layers

    def freeze(self):
        """Freeze module parameters."""

        for param in self.transformer.parameters():
            param.requires_grad = False

    def get_name(self) -> str:
        """Module name.

        Returns:
            string with module name

        """
        return self.name + single_text_hash(self.model_name)

    def get_out_shape(self) -> int:
        """Output shape.

        Returns:
            int with module output shape

        """
        return self.transformer.config.hidden_size
