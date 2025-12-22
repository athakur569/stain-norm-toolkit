"""Macenko stain normalization (Macenko et al., 2009)."""

from __future__ import annotations

import numpy as np

from .base import BaseNormalizer

_OD_THRESHOLD = 0.15
_PERCENTILE = 99


class MacenkoNormalizer(BaseNormalizer):
    """SVD-based stain vector estimation for H&E images.

    Reference:
        Macenko, M. et al. "A method for normalizing histology slides for
        quantitative analysis." ISBI 2009.
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self._he_ref: np.ndarray | None = None
        self._max_ref: np.ndarray | None = None

    def fit(self, image: np.ndarray) -> "MacenkoNormalizer":
        image = self._validate_image(image)
        od = self._rgb_to_od(image)
        he, maxc = self._estimate_stain_vectors(od)
        self._he_ref = he
        self._max_ref = maxc
        self._fitted = True
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        self._check_fitted()
        image = self._validate_image(image)
        h, w = image.shape[:2]
        od = self._rgb_to_od(image)
        he, _ = self._estimate_stain_vectors(od)
        concentrations = np.linalg.lstsq(he, od.T, rcond=None)[0]
        maxc = np.percentile(concentrations, _PERCENTILE, axis=1, keepdims=True)
        concentrations_norm = concentrations * (self._max_ref / maxc)
        od_norm = self._he_ref.dot(concentrations_norm)
        rgb = np.exp(-od_norm).T.reshape(h, w, 3)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------

    def _rgb_to_od(self, image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        img = np.maximum(img, 1)
        od = -np.log(img / 255.0)
        mask = (od >= self.beta).any(axis=1)
        return od[mask]

    def _estimate_stain_vectors(
        self, od: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        _, _, vt = np.linalg.svd(od, full_matrices=False)
        basis = vt[:2].T
        angles = np.arctan2(basis[:, 1], basis[:, 0])
        lo = np.percentile(angles, self.alpha)
        hi = np.percentile(angles, 100 - self.alpha)
        v1 = np.array([np.cos(lo), np.sin(lo)])
        v2 = np.array([np.cos(hi), np.sin(hi)])
        he = np.array([basis.dot(v1), basis.dot(v2)])
        if he[0, 0] < he[1, 0]:
            he = he[[1, 0]]
        concentrations = np.linalg.lstsq(he, od.T, rcond=None)[0]
        maxc = np.percentile(concentrations, _PERCENTILE, axis=1)
        return he, maxc
