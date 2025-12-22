"""Abstract base for stain normalizers."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseNormalizer(ABC):
    """Common interface for all stain normalization methods."""

    def __init__(self) -> None:
        self._fitted = False

    @abstractmethod
    def fit(self, image: np.ndarray) -> "BaseNormalizer":
        """Learn stain statistics from a reference image.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            self (for chaining).
        """

    @abstractmethod
    def transform(self, image: np.ndarray) -> np.ndarray:
        """Normalize a target image to the fitted reference.

        Args:
            image: RGB image (H, W, 3) uint8.

        Returns:
            Normalized RGB image (H, W, 3) uint8.
        """

    def fit_transform(self, reference: np.ndarray, target: np.ndarray) -> np.ndarray:
        """Fit on reference and transform target in one call."""
        return self.fit(reference).transform(target)

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{type(self).__name__} must be fitted before transform()")

    @staticmethod
    def _validate_image(image: np.ndarray) -> np.ndarray:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H, W, 3) image, got shape {image.shape}")
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)
        return image
