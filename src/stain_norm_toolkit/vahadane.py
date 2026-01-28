"""Vahadane stain normalization via sparse NMF (Vahadane et al., 2016)."""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import DictionaryLearning

from .base import BaseNormalizer


class VahadaneNormalizer(BaseNormalizer):
    """Structure-preserving color normalization using sparse NMF.

    Reference:
        Vahadane, A. et al. "Structure-preserving color normalization and
        sparse stain separation for histological images." IEEE TMI 2016.
    """

    def __init__(
        self,
        lambda_: float = 0.1,
        n_iter: int = 20,
        od_threshold: float = 0.15,
    ) -> None:
        super().__init__()
        self.lambda_ = lambda_
        self.n_iter = n_iter
        self.od_threshold = od_threshold
        self._w_ref: np.ndarray | None = None

    def fit(self, image: np.ndarray) -> "VahadaneNormalizer":
        image = self._validate_image(image)
        od = self._get_tissue_od(image)
        self._w_ref = self._get_stain_matrix(od)
        self._fitted = True
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        self._check_fitted()
        image = self._validate_image(image)
        h, w = image.shape[:2]
        od = self._rgb_to_od(image)
        w_src = self._get_stain_matrix(self._get_tissue_od(image))
        h_src = self._get_concentrations(od, w_src)
        h_norm = h_src * (np.percentile(h_src, 99, axis=1, keepdims=True) /
                          (np.percentile(h_src, 99, axis=1, keepdims=True) + 1e-6))
        od_norm = self._w_ref.dot(h_norm)
        rgb = np.exp(-od_norm.T).reshape(h, w, 3)
        return np.clip(rgb * 255, 0, 255).astype(np.uint8)

    # ------------------------------------------------------------------

    def _get_tissue_od(self, image: np.ndarray) -> np.ndarray:
        od = self._rgb_to_od(image)
        mask = np.linalg.norm(od, axis=1) > self.od_threshold
        return od[mask]

    def _get_stain_matrix(self, od: np.ndarray) -> np.ndarray:
        dl = DictionaryLearning(
            n_components=2,
            alpha=self.lambda_,
            max_iter=self.n_iter,
            fit_algorithm="lars",
            transform_algorithm="lasso_lars",
            positive_code=True,
            positive_dict=True,
            random_state=42,
        )
        dl.fit(od)
        w = dl.components_
        # sort so row 0 ≈ haematoxylin (bluish → lower R component)
        if w[0, 0] > w[1, 0]:
            w = w[[1, 0]]
        return w / (np.linalg.norm(w, axis=1, keepdims=True) + 1e-6)

    def _get_concentrations(
        self, od: np.ndarray, w: np.ndarray
    ) -> np.ndarray:
        return np.linalg.lstsq(w.T, od.T, rcond=None)[0]

    @staticmethod
    def _rgb_to_od(image: np.ndarray) -> np.ndarray:
        img = image.reshape(-1, 3).astype(np.float64)
        img = np.maximum(img, 1)
        return -np.log(img / 255.0)
