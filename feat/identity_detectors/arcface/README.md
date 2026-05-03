# ArcFace identity detector

Modern alternative to the default `facenet` identity detector. ArcFace
embeddings disentangle identity from pose and expression much better
than triplet-loss embeddings (FaceNet).

## Usage

```python
from feat import Detector
detector = Detector(identity_model="arcface")
fex = detector.detect("video.mp4")
```

## Model card

- **Architecture:** IR-50 (43.6M params, 166MB safetensors)
- **Training data:** WebFace600K (pseudo-labeled, ~600K identities)
- **Loss:** ArcFace angular margin softmax
- **Source:** Converted from InsightFace's `buffalo_l` pack
  ([releases](https://github.com/deepinsight/insightface/releases))
  via `scripts/convert_arcface_onnx_to_safetensors.py`.
- **Reported benchmarks (TAR @ FAR=1e-4):**
  - LFW: 99.83%
  - IJB-C: 96.18%
  - Compare to `facenet` (VGGFace2): ~80% on IJB-C — ArcFace is
    substantially better on the hard benchmarks where pose and
    expression vary, which is the exact regime of FEAT video data.

## License

The InsightFace **code** is MIT-licensed. The **pretrained weights**
are distributed under InsightFace's "non-commercial research" terms;
see [their model card](https://github.com/deepinsight/insightface) for
the original distribution. Commercial users should validate license
compatibility for their use case. The default `facenet` weights inherit
the same kind of restriction (VGGFace2-trained, also research-only),
so this is not a new license category for py-feat — but it remains
the user's responsibility under the BSD-3-Clause that py-feat itself
ships under.

## Reproducing the conversion

The InsightFace ONNX export folds most BatchNorms into the adjacent
Convs. Our `iresnet.py` matches that fused structure so initializer
names map 1:1 with no surgery. To rebuild the safetensors from
scratch:

```bash
# 1. Download the InsightFace pack (~275MB)
curl -L -o /tmp/buffalo_l.zip \
    https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip
unzip -o /tmp/buffalo_l.zip w600k_r50.onnx -d /tmp/

# 2. Convert + verify
python scripts/convert_arcface_onnx_to_safetensors.py \
    --onnx /tmp/w600k_r50.onnx \
    --out /tmp/arcface_r50.safetensors \
    --backbone r50 \
    --verify
# Expected output: max |diff| < 1e-5, cosine similarity == 1.0

# 3. Upload to py-feat/arcface_r50 on HuggingFace (maintainers only)
huggingface-cli upload py-feat/arcface_r50 /tmp/arcface_r50.safetensors arcface_r50.safetensors
```

For local testing without uploading, set the env var:

```bash
export FEAT_ARCFACE_R50_PATH=/tmp/arcface_r50.safetensors
```

The detector will load the weights from this path instead of fetching
from HuggingFace.
