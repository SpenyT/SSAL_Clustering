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
    @abstractmethod
    def forward(self, X: torch.Tensor) -> torch.Tensor: ...

    @property
    @abstractmethod
    def output_dim(self) -> int: ...

    def fit(
        self, dataset: Dataset, batch_size: int = 256
    ) -> "FeatureExtractor":
        del (
            dataset,
            batch_size,
        )  # literally just so I dont get any warnings from it
        return self

    @torch.no_grad()
    def extract(
        self, dataset: Dataset, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = load_resnet18(
            with_pretrained_weights=pretrained, strip_fc=True
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.backbone(imgs), dim=1)

    @property
    def output_dim(self) -> int:
        return 512


# was using scikit-learn's PCA but was too slow.
# found this source: https://github.com/gngdb/pytorch-pca/blob/main/pca.py
# Thanks @gngdb, absolutely goated
class PCAExtractor(FeatureExtractor):
    def __init__(self, extractor: FeatureExtractor, n_components: int) -> None:
        super().__init__()
        self.extractor = extractor
        self._n_components = n_components

    @staticmethod
    def _svd_flip(
        u: torch.Tensor, v: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sign-flip U and Vt so SVD output is deterministic across runs."""
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


# found this: https://www.reddit.com/r/MachineLearning/comments/1k1nn8d/
# Thought a 60x performance upgrade would be interesting
class UMAPExtractor(FeatureExtractor):
    def __init__(
        self, extractor: FeatureExtractor, n_components: int = 2
    ) -> None:
        super().__init__()
        self.extractor = extractor
        self._n_components = n_components
        self._umap = None

    def fit(self, dataset: Dataset, batch_size: int = 256) -> "UMAPExtractor":
        features, _, _ = self.extractor.extract(dataset, batch_size)
        if HAS_CUML:
            from cuml.manifold.umap import UMAP as _UMAP
        else:
            from umap import UMAP as _UMAP
        self._umap = _UMAP(n_components=self._n_components)
        self._umap.fit(features)
        return self

    def extract(
        self, dataset: Dataset, batch_size: int = 256
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._umap is None:
            raise RuntimeError("Call fit() first (UMAPExtractor)")
        features, indices, labels = self.extractor.extract(dataset, batch_size)
        if HAS_CUML:
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
