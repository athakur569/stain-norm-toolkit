"""Reinhard color normalization (Reinhard et al., 2001)."""

from __future__ import annotations

import cv2
import numpy as np

from .base import BaseNormalizer


class ReinhardNormalizer(BaseNormalizer):
    """Lab color space mean/std transfer.

    Reference:
        Reinhard, E. et al. "Color transfer between images."
        IEEE CG&A 2001.
    """

    def __init__(self) -> None:
        super().__init__()
        self._mean_ref: np.ndarray | None = None
        self._std_ref: np.ndarray | None = None

    def fit(self, image: np.ndarray) -> "ReinhardNormalizer":
        image = self._validate_image(image)
        lab = self._to_lab(image)
        self._mean_ref = lab.mean(axis=(0, 1))
        self._std_ref = lab.std(axis=(0, 1)) + 1e-6
        self._fitted = True
        return self

    def transform(self, image: np.ndarray) -> np.ndarray:
        self._check_fitted()
        image = self._validate_image(image)
        lab = self._to_lab(image)
        mean = lab.mean(axis=(0, 1))
        std = lab.std(axis=(0, 1)) + 1e-6
        lab_norm = (lab - mean) / std * self._std_ref + self._mean_ref
        return self._from_lab(np.clip(lab_norm, 0, 255).astype(np.uint8))

    @staticmethod
    def _to_lab(image: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab).astype(np.float64)

    @staticmethod
    def _from_lab(lab: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(lab, cv2.COLOR_Lab2BGR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
