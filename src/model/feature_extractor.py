from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.amp import autocast
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from glob_config import (
    DEVICE,
    NUM_WORKERS,
    PIN_MEMORY,
    SEED,
    USE_AMP,
    seed_worker,
    HAS_CUML,
)
from model.resnet import load_resnet18


class FeatureExtractor(nn.Module, ABC):
    """
    Abstract base class for all feature extractors.

    Defines the shared interface and extraction pipeline for all
    feature extractors (e.g. ResnetExtractor, PCAExtractor). Not intended
    to be used directly: always subclass and implement the forward() and
    output_dim methods.

    Arguments
    ---------
    None
        This class takes no contructor arguments. Arguments are handled by
        each concrete subclass.

    Example
    -------
    >>> # Use a concrete subclass, not FeatureExtractor directly
    >>> extractor = ResnetExtractor(pretrained=True)
    >>> features, indices, labels = extractor.extract(dataset, batch_size=256)
    """

    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Map a batch of images to their feature embeddings.

        Must be implemented by all concrete subclasses.

        Arguments
        ---------
        X: torch.Tensor
            A batch of images, shape (B, C, H, W).

        Returns
        -------
        torch.Tensor
            The corresponding feature embeddings, shape (B, output_dim).
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """
        The dimensionality of the feature embeddings produced by this
        extractor.

        Must be implemented by all concrete subclasses.

        Arguments
        ---------
        None
            This is a property, not a method, so it takes no arguments.

        Returns
        -------
        int
            The output dimensionality of the feature embeddings.
        """

    def fit(
        self, _dataset: Dataset, _batch_size: int = 256
    ) -> "FeatureExtractor":
        """
        Fit the feature extractor to the given dataset.

        By default, this method does nothing and just returns self.
        Overriden by subclasses that require fitting
        (e.g. PCAExtractor, UMAPExtractor).

        Arguments
        ---------
        dataset: Dataset
            The dataset to fit the feature extractor on.
        batch_size: int
            The batch size to use when extracting features for fitting.

        Returns
        -------
        FeatureExtractor
            The fitted feature extractor (self).
        """
        return self

    @torch.no_grad()
    def extract(
        self, dataset: Dataset, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Extract features for the entire dataset.

        This method runs the dataset through the feature extractor in batches
        and collects the resulting features, indices, and labels. It handles
        all the necessary DataLoader setup and device management.

        Arguments
        ---------
        dataset: Dataset
            The dataset to extract features from.
        batch_size: int
            The batch size to use when extracting features.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray]:
            - features: shape (N, output_dim), the extracted feature embeddings
                        for all N samples.
            - indices: shape (N,), the original dataset indices corresponding
                        to each feature embedding.
            - labels: shape (N,), the labels for each sample in the dataset.
        """
        self.eval()
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=NUM_WORKERS > 0,
            prefetch_factor=2 if NUM_WORKERS > 0 else None,
            worker_init_fn=seed_worker,
            generator=torch.Generator().manual_seed(SEED),
        )

        all_features, all_indices, all_labels = [], [], []

        for (imgs, labels), idxs in tqdm(
            loader, desc="Extracting features", leave=False
        ):
            imgs = imgs.to(DEVICE, non_blocking=True)
            if USE_AMP:
                with autocast(device_type=DEVICE.type):
                    feats = self(imgs)
            else:
                feats = self(imgs)

            all_features.append(feats.cpu().float())
            all_indices.append(idxs)
            all_labels.append(labels)

        features = torch.cat(all_features).numpy()
        indices = torch.cat(all_indices).numpy()
        labels = torch.cat(all_labels).numpy()
        return features, indices, labels


class ResnetExtractor(FeatureExtractor):
    """
    Feature extractor backed by a ResNet-18 encoder.

    Passes images through a ResNet-18 backbone (without its ) and
    returns normalized embeddings (I used L2 norm) of dimension 512.
    Optionally initialized with ImageNet-pretrained weights.

    Arguments
    ---------
    pretrained : bool
        If True, loads ImageNet-pretrained weights into the backbone.
        Default: True.

    Example
    -------
    >>> extractor = ResnetExtractor(pretrained=True)
    >>> features, indices, labels = extractor.extract(dataset, batch_size=256)
    >>> features.shape
    (1000, 512)
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = load_resnet18(
            with_pretrained_weights=pretrained, strip_fc=True
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(imgs), dim=1)

    @property
    def output_dim(self) -> int:
        return 512  # Resnet is always has 512 dimensions before final FC layer


class PCAExtractor(FeatureExtractor):
    """
    Feature extractor that applies PCA on top of a base extractor.

    Wraps another FeatureExtractor and reduces its output dimensionality
    via a GPU-accelerated PCA (SVD-based). Must be fitted on a dataset
    before use. Implements sklearn-style fit/transform using PyTorch
    to avoid the performance cost of scikit-learn's CPU-bound PCA.

    References
    ----------
    - PCA implemented via full SVD on GPU:
        https://github.com/gngdb/pytorch-pca/blob/main/pca.py

    Arguments
    ---------
    extractor : FeatureExtractor
        Base extractor whose embeddings will be reduced.
    n_components : int
        Number of principal components to keep.

    Example
    -------
    >>> base = ResnetExtractor(pretrained=True)
    >>> pca = PCAExtractor(base, n_components=64)
    >>> pca.fit(dataset)
    >>> features, indices, labels = pca.extract(dataset, batch_size=256)
    >>> features.shape
    (1000, 64)
    """

    def __init__(self, extractor: FeatureExtractor, n_components: int) -> None:
        super().__init__()
        self.extractor = extractor
        self._n_components = n_components

    @staticmethod
    def _svd_flip(
        u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sign-flip U and Vt to make SVD output deterministic across runs.

        Resolves the sign ambiguity in SVD by ensuring the largest
        absolute value in each column of U is always positive.

        Arguments
        ---------
        u : torch.Tensor
            Left singular vectors, shape (N, K).
        v : torch.Tensor
            Right singular vectors, shape (K, D).

        Returns
        -------
        u : torch.Tensor
            Sign-corrected left singular vectors, shape (N, K).
        v : torch.Tensor
            Sign-corrected right singular vectors, shape (K, D).
        """
        max_abs_cols = torch.argmax(torch.abs(u), dim=0)
        signs = torch.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, None]
        return u, v

    def fit(self, dataset: Dataset, batch_size: int = 256) -> "PCAExtractor":
        features, _, _ = self.extractor.extract(dataset, batch_size)
        return self._fit(torch.from_numpy(features).to(DEVICE))

    @torch.no_grad()
    def _fit(self, X: torch.Tensor) -> "PCAExtractor":
        """
        Compute and store PCA components from a pre-extracted feature matrix.

        Centers the data, runs full SVD, applies sign correction for
        determinism, and registers the mean and top-K components as
        buffers so they are saved with the model state.

        Arguments
        ---------
        X : torch.Tensor
            Feature matrix of shape (N, D), where N is the number of
            samples and D is the feature dimensionality.

        Returns
        -------
        PCAExtractor
            The fitted extractor (self), for method chaining.
        """
        self.register_buffer("mean_", X.mean(0, keepdim=True))
        U, _, Vt = torch.linalg.svd(X - self.mean_, full_matrices=False)
        U, Vt = self._svd_flip(U, Vt)
        self.register_buffer("components_", Vt[: self._n_components])
        return self

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        assert hasattr(self, "components_"), "Call fit() before forward()"
        feats = self.extractor(imgs)
        return nn.functional.normalize(
            torch.matmul(feats - self.mean_, self.components_.t()), dim=1
        )

    @property
    def output_dim(self) -> int:
        return self._n_components


class UMAPExtractor(FeatureExtractor):
    """
    Feature extractor that applies UMAP on top of a base extractor.

    Wraps another FeatureExtractor and reduces its output dimensionality
    via UMAP. Uses cuML's GPU-accelerated UMAP if available, otherwise
    falls back to the CPU-based umap-learn implementation.
    UMAP is not differentiable and does not implement forward(): always
    use extract() directly.

    References
    ----------
    - UMAP using cuML for GPU acceleration:
        https://www.reddit.com/r/MachineLearning/comments/1k1nn8d/

    Arguments
    ---------
    extractor : FeatureExtractor
        Base extractor whose embeddings will be reduced.
    n_components : int
        Number of UMAP dimensions to reduce to. Default: 2.

    Example
    -------
    >>> base = ResnetExtractor(pretrained=True)
    >>> umap = UMAPExtractor(base, n_components=2)
    >>> umap.fit(dataset)
    >>> features, indices, labels = umap.extract(dataset, batch_size=256)
    >>> features.shape
    (1000, 2)
    """

    def __init__(
        self, extractor: FeatureExtractor, n_components: int = 2
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self._n_components = n_components
        self._umap = None
        self._use_cuml = False

    def fit(self, dataset: Dataset, batch_size: int = 256) -> "UMAPExtractor":
        features, _, _ = self.extractor.extract(dataset, batch_size)
        if HAS_CUML:
            from cuml.manifold.umap import UMAP as _UMAP  # type: ignore

            self._use_cuml = True
        else:
            from umap import UMAP as _UMAP  # type: ignore

            self._use_cuml = False
        self._umap = _UMAP(n_components=self._n_components)
        self._umap.fit(features)
        return self

    def extract(
        self, dataset: Dataset, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._umap is None:
            raise RuntimeError("Call fit() first (UMAPExtractor)")
        features, indices, labels = self.extractor.extract(dataset, batch_size)
        if self._use_cuml:
            import cupy as cp  # type: ignore

            reduced = cp.asnumpy(self._umap.transform(cp.asarray(features)))
        else:
            reduced = self._umap.transform(features)
        return reduced, indices, labels

    def forward(self, _: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("UMAPExtractor use extract()")

    @property
    def output_dim(self) -> int:
        return self._n_components
