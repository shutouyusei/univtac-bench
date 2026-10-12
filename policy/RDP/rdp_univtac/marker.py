"""Fingertip marker motion -> RDP's 15-D tactile embedding.

RDP (official code, GelSight Mini) tracks a 7 x 9 marker grid, takes each marker's offset from its reference
position normalised by the image size (x / width, y / height), flattens the 63 offsets row-major to 126 numbers
(x0, y0, x1, y1, ...) and embeds them with a fixed PCA, ``emb = (x - mean) @ W`` with W of shape (126, 15)
(``model/common/pca_embedding.py``). UniVTAC's simulated GelSight Mini records the same 63-marker grid as
``tactile/<side>_tactile/marker`` with shape (T, 2, 63, 2) = [reference position, current position] in pixels of
its 320 x 240 image. The PCA is fit here with numpy (sklearn's PCA: centred data, top right singular vectors).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

IMAGE_SIZE = (320, 240)  # UniVTAC gsmini tactile image (width, height), envs/sensors/tactile.py


def displacement(marker: np.ndarray, image_size: tuple[int, int] = IMAGE_SIZE) -> np.ndarray:
    """(T, 2, N, 2) [reference, current] marker positions -> (T, 2N) normalised offsets x0, y0, x1, y1, ..."""
    marker = np.asarray(marker)
    if marker.ndim != 4 or marker.shape[1] != 2 or marker.shape[3] != 2:
        raise ValueError(f"expected (T, 2, N, 2) [reference, current] marker positions, got {marker.shape}")
    offset = (marker[:, 1] - marker[:, 0]) / np.asarray(image_size, dtype=np.float64)
    return offset.reshape(len(marker), -1).astype(np.float32)


@dataclass
class MarkerPCA:
    mean: np.ndarray  # (D,)
    components: np.ndarray  # (D, k), columns = principal axes, decreasing variance

    def embed(self, x: np.ndarray) -> np.ndarray:
        return ((np.asarray(x, dtype=np.float64) - self.mean) @ self.components).astype(np.float32)

    def save(self, path: Path) -> None:
        np.savez(path, mean=self.mean, components=self.components)

    @classmethod
    def load(cls, path: Path) -> MarkerPCA:
        with np.load(path) as f:
            return cls(mean=f["mean"], components=f["components"])


def fit_pca(x: np.ndarray, n_components: int = 15) -> MarkerPCA:
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return MarkerPCA(mean=mean, components=vt[:n_components].T.copy())
