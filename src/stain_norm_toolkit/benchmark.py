"""Benchmark stain normalizers across a set of source/target patches."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from . import MacenkoNormalizer, ReinhardNormalizer, VahadaneNormalizer

_METHODS = {
    "macenko": MacenkoNormalizer,
    "reinhard": ReinhardNormalizer,
    "vahadane": VahadaneNormalizer,
}


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _psnr(original: np.ndarray, normalized: np.ndarray) -> float:
    mse = np.mean((original.astype(float) - normalized.astype(float)) ** 2)
    if mse == 0:
        return float("inf")
    return 20 * np.log10(255.0 / np.sqrt(mse))


def run_benchmark(
    source_dir: str | Path,
    target_dir: str | Path,
    methods: list[str],
    output_dir: str | Path,
    reference_image: str | Path | None = None,
) -> dict:
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source_paths = sorted(source_dir.glob("*.png")) + sorted(source_dir.glob("*.jpg"))
    if not source_paths:
        raise FileNotFoundError(f"No images found in {source_dir}")

    # Use first target image as reference if not specified
    target_paths = sorted(target_dir.glob("*.png")) + sorted(target_dir.glob("*.jpg"))
    ref_path = Path(reference_image) if reference_image else target_paths[0]
    ref_img = _load_image(ref_path)

    results = {}
    for method_name in methods:
        if method_name not in _METHODS:
            raise ValueError(f"Unknown method: {method_name}. Choose from {list(_METHODS)}")

        normalizer = _METHODS[method_name]()
        normalizer.fit(ref_img)

        psnrs, times = [], []
        out_method_dir = output_dir / method_name
        out_method_dir.mkdir(exist_ok=True)

        for src_path in source_paths:
            src = _load_image(src_path)
            t0 = time.perf_counter()
            normalized = normalizer.transform(src)
            elapsed = time.perf_counter() - t0
            psnrs.append(_psnr(src, normalized))
            times.append(elapsed)
            Image.fromarray(normalized).save(out_method_dir / src_path.name)

        results[method_name] = {
            "mean_psnr": float(np.mean(psnrs)),
            "std_psnr": float(np.std(psnrs)),
            "mean_time_s": float(np.mean(times)),
            "n_images": len(source_paths),
        }
        print(f"{method_name:12s}  PSNR {np.mean(psnrs):.2f} ± {np.std(psnrs):.2f} dB  "
              f"time {np.mean(times)*1000:.1f} ms/patch")

    with open(output_dir / "benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_dir / 'benchmark_results.json'}")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Benchmark stain normalization methods")
    parser.add_argument("--source", required=True, help="Directory of source patches")
    parser.add_argument("--target", required=True, help="Directory of target patches")
    parser.add_argument("--methods", nargs="+", default=["macenko", "reinhard", "vahadane"])
    parser.add_argument("--output", default="benchmark_results/")
    parser.add_argument("--reference", default=None, help="Reference image path")
    args = parser.parse_args()

    run_benchmark(args.source, args.target, args.methods, args.output, args.reference)
