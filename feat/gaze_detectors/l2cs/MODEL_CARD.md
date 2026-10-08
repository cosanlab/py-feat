---
library_name: py-feat
pipeline_tag: image-classification
tags:
- gaze-estimation
license: other
license_name: mit-code-with-gaze360-research-terms
license_link: https://github.com/cosanlab/py-feat/blob/main/LICENSES/PYFEAT-V1.md
---

# L2CS-Net (Gaze Estimation)

## Model Description

L2CS-Net regresses gaze direction (pitch, yaw) from a face crop. It
formulates gaze estimation as a 90-bin classification problem over each
axis (4°/bin resolution covering [-180°, +180°]), then computes the
expected value across bins for a continuous angle output. ResNet50
backbone, two parallel FC heads.

Reported accuracy (from the original L2CS-Net paper):
- Gaze360 test split: ~3.92° MAE
- MPIIFaceGaze leave-one-out: ~4.16° MAE

These are state-of-the-art numbers for gaze-from-face-crop estimation
(2022); for context, geometric iris-eye approaches typically land at
8-15° MAE on the same benchmarks.

## Model Details

- **Architecture**: ResNet50 + dual classification heads (2 × 90-bin)
- **Input**: 224 × 224 RGB face crop, ImageNet-normalized
- **Output**: pitch, yaw in radians (head-centric frame)
- **Bin resolution**: 4° per bin, covering [-180°, +180°]
- **Backbones available**: ResNet50 (default), ResNet18 (lighter)
- **Framework**: PyTorch (port of upstream MIT code)

## Training data (upstream)

- **Gaze360** (Kellnhofer et al., 2019): in-the-wild gaze annotations
  with 360° head pose coverage. ~127k images.
- **MPIIFaceGaze** (Zhang et al., 2017): unconstrained office captures
  with screen-targeted gaze ground truth. ~213k images.

The upstream maintainer trains separate checkpoints for each dataset.
The checkpoint shipped by Py-Feat is **Gaze360-trained**:
`l2cs_gaze360_resnet50.safetensors`, converted from
`L2CSNet_gaze360.pkl`. MPIIFaceGaze is a separate upstream training and
evaluation configuration, not a second training corpus for this checkpoint.

## Model Sources

- **Original repository (MIT)**: [Ahmednull/L2CS-Net](https://github.com/Ahmednull/L2CS-Net)
- **Paper**: [arXiv:2203.03339](https://arxiv.org/abs/2203.03339)
- **Pretrained weights (upstream)**: Google Drive folder linked from the
  upstream README (`L2CSNet_gaze360.pkl`). Py-Feat hosts a re-packaged
  `.safetensors` version on this repo to avoid the pickle (`.pkl`)
  deserialization path on user machines. The conversion is documented
  at `scripts/convert_l2cs_pickle_to_safetensors.py` in the py-feat
  repo; no architecture or weight values are modified.

## Acknowledgements

This distribution is a **re-host** of the official L2CS-Net weights
trained by Abdelrahman, Hempel, Khalifa, and Al-Hamadi (Otto-von-Guericke
University Magdeburg). All training, hyperparameter selection, and
benchmark numbers are credited to the original authors. The py-feat
project provides only:
- A PyTorch port of the inference code at `feat/gaze_detectors/l2cs/l2cs_model.py`
- This re-packaged `.safetensors` artifact for downstream safety
- Integration with `feat.Detectorv1` and `feat.MPDetector`'s pipelines

The shipped checkpoint's training data are credited to:
- Gaze360 — Kellnhofer, Recasens, Stent, Matusik, Torralba (MIT)

The separate MPIIFaceGaze configuration uses data from Zhang, Sugano,
Fritz, and Bulling (MPI Saarbrücken).

## Citation

```bibtex
@article{l2csnet2022,
  title={L2CS-Net: Fine-Grained Gaze Estimation in Unconstrained Environments},
  author={Abdelrahman, Ahmed and Hempel, Thorsten and Khalifa, Aly and
          Al-Hamadi, Ayoub},
  journal={arXiv preprint arXiv:2203.03339},
  year={2022}
}
```

## License

The original implementation is [MIT-licensed](https://github.com/Ahmednull/L2CS-Net/blob/main/LICENSE).
The historical model-card designation was MIT. The current `other` label
links this scope notice to distinguish MIT code from the separate Gaze360
research terms; it is not a new combined license or a revocation of existing
valid grants. Conversion to safetensors does not create additional rights.

The shipped checkpoint's [Gaze360 source license](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md)
expressly excludes commercial applications of models trained on the dataset.
Its restrictions on use and distribution require separate assessment of
public weight sharing. The MIT implementation license does not waive these
express conditions, and this clarification does not revoke
existing valid grants.

See the [v1 component notice](https://github.com/cosanlab/py-feat/blob/main/LICENSES/PYFEAT-V1.md)
and [dataset register](https://github.com/cosanlab/py-feat/blob/main/LICENSES/DATASETS.md)
for checkpoint provenance and the outstanding permission review. Terms for
the separate MPIIFaceGaze models must be assessed for those checkpoints.

The applicable acquisition terms and any separate permissions must be
checked for the exact checkpoint; today's source notice does not establish
the history of every upstream grant.
