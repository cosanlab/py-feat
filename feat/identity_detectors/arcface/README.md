# ArcFace identity detector

The default identity detector in py-feat ≥ 0.7. ArcFace embeddings
disentangle identity from pose and expression much better than the
prior FaceNet/triplet-loss embeddings — exactly what FEAT video
clustering needs.

For the full model card (training data, benchmarks, limitations,
license details, citations) see
**[`MODEL_CARD.md`](./MODEL_CARD.md)**. That file is also uploaded
verbatim as the README on the HuggingFace repo
[`py-feat/arcface_r50`](https://huggingface.co/py-feat/arcface_r50).

## Quick reference

| | |
|---|---|
| Architecture | IResNet-50 (43.6 M params) |
| Embedding dim | 512 |
| Source | InsightFace `buffalo_l` (file `w600k_r50.onnx`) |
| Training data | WebFace600K (Tsinghua) |
| File size | 166 MB safetensors |
| File hosted at | `py-feat/arcface_r50` on HuggingFace |
| Inference cost vs. FaceNet | +4 % per frame (13.0 → 13.5 ms/frame on M5 MBP MPS) |

## Usage

```python
from feat import Detectorv1

# Default in v0.7+
detector = Detectorv1()

# Or explicit:
detector = Detectorv1(identity_model="arcface")

# To pin the prior FaceNet for backwards compatibility:
detector = Detectorv1(identity_model="facenet")
```

## License at a glance

- **InsightFace code** (the architecture, training scripts): MIT.
- **The pretrained weights we distribute**: non-commercial research
  only (InsightFace's terms + WebFace600K's terms, both layered).
- **Py-Feat's integration code in this directory**: MIT (same as the
  rest of py-feat).
- **The conversion script** (`scripts/convert_arcface_onnx_to_safetensors.py`):
  MIT — but the converted weights it produces inherit the upstream
  restriction.

The full attribution and citation block, including BibTeX entries for
the ArcFace paper, InsightFace project, WebFace260M paper, and the
ResNet method paper, is in [`MODEL_CARD.md`](./MODEL_CARD.md). The
Py-Feat root [`LICENSE`](../../../LICENSE) lists this model in the
third-party-licenses section.

## Reproducing the conversion

```bash
# 1. Pull buffalo_l (~275 MB) from InsightFace's GitHub release
curl -L -o /tmp/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip w600k_r50.onnx -d /tmp/

# 2. Convert + verify (numerical equivalence to source ONNX)
python scripts/convert_arcface_onnx_to_safetensors.py \
    --onnx /tmp/w600k_r50.onnx \
    --out  /tmp/arcface_r50.safetensors \
    --backbone r50 \
    --verify
# Expected: max |diff| < 1e-5, cosine similarity = 1.0

# 3. (maintainers only) Upload to HuggingFace, including the model card
huggingface-cli upload py-feat/arcface_r50 \
    feat/identity_detectors/arcface/MODEL_CARD.md README.md
huggingface-cli upload py-feat/arcface_r50 \
    /tmp/arcface_r50.safetensors arcface_r50.safetensors
```

## Local testing without HF upload

```bash
export FEAT_ARCFACE_R50_PATH=/tmp/arcface_r50.safetensors
```

The detector will load weights from this path instead of fetching from
HuggingFace.
