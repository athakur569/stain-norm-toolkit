# stain-norm-toolkit

Stain normalization for H&E whole-slide images: Macenko, Reinhard, and Vahadane methods with a unified benchmarking interface.

## Methods
- **Macenko** — SVD-based stain vector estimation (Macenko et al., 2009)
- **Reinhard** — Lab color space mean/std transfer (Reinhard et al., 2001)
- **Vahadane** — Sparse NMF stain separation (Vahadane et al., 2016)

## Installation
```bash
pip install stain-norm-toolkit
```

## Quick start
```python
from stain_norm_toolkit import MacenkoNormalizer, ReinhardNormalizer

normalizer = MacenkoNormalizer()
normalizer.fit(reference_image)           # numpy RGB array (H, W, 3)
normalized = normalizer.transform(target) # same shape
```

## Benchmark
```bash
python -m stain_norm_toolkit.benchmark \
  --source patches/source/ \
  --target patches/target/ \
  --methods macenko reinhard vahadane \
  --output results/
```

## License
MIT
