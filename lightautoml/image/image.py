"""
Image feature extractors
"""

from copy import copy
from typing import List, Callable, Sequence, Optional, Any, Union

import cv2
import numpy as np
import torch
import torch.nn as nn
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch import ToTensorV2
from efficientnet_pytorch import EfficientNet
from joblib import Parallel, delayed
from log_calls import record_history
from sklearn.base import TransformerMixin
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import pil_loader
from ..text.utils import seed_everything, parse_devices

numeric = Union[int, float]


@record_history(enabled=False)
class ColorFeatures:
    """Basic class to compute color features."""

    def __init__(self, hist_size: int = 30, is_hsv: bool = True):
        """Create normalized color histogram for rgb or hsv image.

        Works with RGB and grayscale images only.

        Args:
            hist_size: number of bins for each channel.
            is_hsv: convert image to hsv.

        """
        self.hist_size = hist_size
        self.is_hsv = is_hsv
        self._f_names = ['h', 's', 'v'] if self.is_hsv else ['r', 'g', 'b']

    def compute_histogram(self, img: np.ndarray) -> List[numeric]:
        """Compute normalized color histogram for one channel.

        Args:
            img: image np.ndarray with shape (h, w).

        Returns:
            list of channel histogram values.

        """
        # TODO: add value range check
        hist = cv2.calcHist([img], [0],
                            mask=None,
                            histSize=[self.hist_size],
                            ranges=(0, 255))[:, 0]

        return list(hist / hist.sum())

    def get_names(self) -> List[str]:
        """Define how to get features names list.

        Returns:
            list of features names.

        """
        return [j for i in [['color_' + j + '_' + str(i) for i in np.arange(self.hist_size)] for j in self._f_names] for j in i]

    def get_features(self, img: np.ndarray) -> List[numeric]:
        """Calculate normalized color histogram for rgb or hsv image.

        Args:
            img: np.ndarray with shape (h, w).

        Returns:
            list of histogram values.

        """
        img = copy(img)
        if self.is_hsv:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

        res = []
        for i in range(3):
            channel = img[:, :, i]
            res.extend(self.compute_histogram(channel))

        return res


@record_history(enabled=False)
class CreateImageFeatures:
    """Class for parallel histogram computation."""

    def __init__(self, hist_size: int = 30, is_hsv: bool = True, n_jobs: int = 4, loader: Callable = pil_loader):
        """Create normalized color histogram for rgb or hsv image.

        Args:
            hist_size: number of bins for each channel.
            is_hsv: convert image to hsv.
            n_jobs: number of threads for multiprocessing.
            loader: callable for reading image from path.

        """
        self.fe = ColorFeatures(hist_size, is_hsv)
        self.n_jobs = n_jobs
        self.loader = loader

    def process(self, im_path_i: str) -> List[numeric]:
        """Create normalized color histogram for input image by its path.

        Args:
            im_path_i: str with path to the image.

        Returns:
            list of histogram values.

        """
        im_i = np.array(self.loader(im_path_i))
        if len(im_i.shape) == 2:
            im_i = cv2.cvtColor(im_i, cv2.COLOR_GRAY2RGB)
        res_i = self.fe.get_features(im_i)
        return res_i

    def transform(self, samples: Sequence[str]) -> np.ndarray:
        """Transform input sequence with paths to histogram values.

        Args:
            samples: sequence with images paths.

        Returns:
            array of histograms.

        """
        res = Parallel(self.n_jobs)(delayed(self.process)(im_path_i) for im_path_i in samples)
        return np.vstack(res)


@record_history(enabled=False)
class EffNetImageEmbedder(nn.Module):
    """Class to compute EfficientNet embeddings."""

    def __init__(self, model_name: str = 'efficientnet-b0', weights_path: Optional[str] = None, is_advprop: bool = True,
                 device=torch.device('cuda:0')):
        """Pytorch module for image embeddings based on efficientnet model.

        Args:
            model_name: name of effnet model.
            weights_path: path to saved weights.
            is_advprop: use adversarial training.
            devices: device to use.

        """
        super(EffNetImageEmbedder, self).__init__()
        self.device = device
        self.model = EfficientNet.from_pretrained(model_name, weights_path=weights_path, advprop=is_advprop,
                                                  include_top=False).eval().to(self.device)
        self.feature_shape = self.get_shape()
        self.is_advprop = is_advprop
        self.model_name = model_name

    @torch.no_grad()
    def get_shape(self) -> int:
        """Calculate output embedding shape.

        Returns:
            int shape of embedding.

        """
        return self.model(torch.randn(1, 3, 224, 224).to(self.device)).squeeze().shape[0]

    def forward(self, x) -> torch.Tensor:
        out = self.model(x)
        return out[:, :, 0, 0]


@record_history(enabled=False)
class ImageDataset:
    """Image Dataset Class."""

    def __init__(self, data: Sequence[str], is_advprop: bool = True, loader: Callable = pil_loader):
        """Pytorch Dataset for EffNetImageEmbedder.

        Args:
            data: sequence of paths.
            is_advprop: use adversarial training.
            loader: callable for reading image from path.

        """
        self.X = data
        self.transforms = Compose([Resize(224, 224),
                                   Normalize([0.5] * 3, [0.5] * 3) if is_advprop else Normalize(),
                                   ToTensorV2()
                                   ])
        self.loader = loader

    def __getitem__(self, idx: int) -> np.ndarray:
        path = self.X[idx]
        img = np.array(self.loader(path))
        img = self.transforms(image=img)['image']
        return img

    def __len__(self):
        return len(self.X)


@record_history(enabled=False)
class DeepImageEmbedder(TransformerMixin):
    """Transformer for image embeddings."""

    def __init__(self, device: torch.device = torch.device('cuda:0'), n_jobs=4, random_state=42, is_advprop=True,
                 model_name='efficientnet-b0', weights_path: Optional[str] = None, batch_size: int = 128, verbose: bool = True):
        """Pytorch Dataset for EffNetImageEmbedder.

        Args:
            device: torch device.
            n_jobs: number of threads for dataloader.
            random_state: seed.
            is_advprop: use adversarial training.
            model_name: name of effnet model.
            weights_path: path to saved weights.
            batch_size: batch size.
            verbose: verbose data processing.

        """
        super(DeepImageEmbedder, self).__init__()
        assert model_name in {f'efficientnet-b{i}' for i in range(8)}

        self.device, self.device_ids = parse_devices(device)
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.is_advprop = is_advprop
        self.batch_size = batch_size
        self.verbose = verbose
        seed_everything(random_state)

        self.model = EffNetImageEmbedder(model_name, weights_path, self.is_advprop, self.device)

    def fit(self, data: Any = None):
        return self

    @torch.no_grad()
    def transform(self, data: Sequence[str]) -> np.ndarray:
        """Calculate image embeddings from pathes.

        Args:
            data: sequence of paths.

        Returns:
            array of embeddings.

        """

        data = ImageDataset(data, self.is_advprop)
        loader = DataLoader(data, batch_size=self.batch_size, shuffle=False, num_workers=self.n_jobs)

        result = []
        if self.verbose:
            loader = tqdm(loader)

        self.model = self.model.to(self.device)
        for batch in loader:
            embed = self.model(batch.float().to(self.device)).detach().cpu().numpy()
            result.append(embed.astype(np.float32))

        return np.vstack(result)
