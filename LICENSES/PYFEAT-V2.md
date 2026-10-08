# Py-Feat v2.0 licensing

This document covers the multitask `Detectorv2` family. The detailed dataset
inventory refers to `face_multitask_v28.safetensors` in
[`py-feat/face_multitask_v2`](https://huggingface.co/py-feat/face_multitask_v2),
not every older checkpoint in that repository. Reviewed October 8, 2026.

## Software

Py-Feat's original source code and documentation remain under the
[MIT License](../LICENSE), including commercial use. Third-party code
retains its own notices. The noncommercial restrictions described below
concern particular pretrained assets and data permissions, not a blanket
restriction on the Py-Feat software.

## Pretrained weights and upstream components

The multitask weights have been designated for noncommercial research.
They cannot be represented as MIT-only:

- The backbone is initialized from
  [`convnextv2_tiny.fcmae_ft_in22k_in1k`](https://huggingface.co/timm/convnextv2_tiny.fcmae_ft_in22k_in1k),
  whose pretrained weights are **CC BY-NC 4.0**. The
  [ConvNeXt-V2 code license](https://github.com/facebookresearch/ConvNeXt-V2#license)
  is MIT and explicitly distinguishes the ImageNet weights. The multitask
  checkpoint modifies the pretrained network through joint fine-tuning and
  adds task heads. Preserve the upstream attribution and license notice.
- [Gaze360's research license](https://github.com/erkil1452/gaze360/blob/master/LICENSE.md)
  expressly excludes commercial applications of models trained on its data.
  Its limits on use and redistribution also require review for public
  sharing. ETH-XGaze's access conditions expressly prohibit training models
  for commercial products. These are additional restrictions beyond the
  backbone license.
- Geometry targets are distilled from MediaPipe and
  [img2pose](https://github.com/vitoralbiero/img2pose/blob/main/license.md).
  img2pose is CC BY-NC 4.0. Teacher licensing and permission to distribute
  student weights require separate consideration; distillation is not a
  blanket permission to relicense.
- The AU architecture adapts
  [ME-GraphAU](https://github.com/CVI-SZU/ME-GraphAU). Its upstream source
  terms must be retained for any copied code; architecture attribution does
  not establish a license for checkpoints or training data.
- RetinaFace and optional identity weights are separate downloads with
  their own [component notices](PYFEAT-V1.md). ArcFace is not the source of
  the multitask network's licensing restrictions. Disabling identity
  extraction does not remove the network's own upstream restrictions.

**A research-only label is not, by itself, authorization for unrestricted
public distribution.** Aff-Wild2's newer agreement explicitly regulates
trained-model sharing; BP4D and BP4D+ contain provisions concerning
parameter sets and algorithms. Their applicability and any necessary
permission must be resolved for the particular checkpoint. This notice
does not grant rights beyond those available from the relevant holders.

## Commercial-use inquiries

The current multitask checkpoints are not offered as commercially cleared
weights. Py-Feat does not broker third-party permissions or offer an
additional commercial sublicense that overrides upstream restrictions.
The MIT software grant and existing valid component licenses are unchanged.

Users seeking commercial use must assess the exact assets and applicable
terms, and obtain any necessary permissions directly from the relevant
rights holders. The [dataset register](DATASETS.md) links provider sources;
the backbone and teacher sources are listed above. A dataset download
approval is not necessarily permission for commercial use of an already
trained checkpoint. Py-Feat cannot certify that independently obtained
permissions resolve every applicable right.

Alternatively, use independently cleared weights with the MIT software.
The [commercial retraining table](COMMERCIAL-RETRAINING.md) identifies the
remaining work for a future release. Downstream permissions do not by
themselves resolve the distributor's own agreement obligations or weight
sharing authority.

## Current checkpoint's dataset roles

The current manuscript's training tables distinguish these sources:

| Use | Corpora |
|---|---|
| Stage 1 geometry distillation | CelebV-HQ, BP4D, BP4D+, DISFA, EmotioNet, CK+, UNBC-McMaster Shoulder Pain |
| Stages 2-3 AU supervision | BP4D, BP4D+, DISFA, EmotioNet, CK+, UNBC-McMaster Shoulder Pain, AM-FED, Aff-Wild2 AU |
| Stages 2-3 emotion supervision | AffectNet, RAF-DB, FER+, ExpW, MELD |
| Stages 2-3 valence/arousal supervision | AffectNet, Aff-Wild2 V/A, AFEW-VA |
| Stages 2-3 gaze supervision | ETH-XGaze, Gaze360, MPIIGaze, Columbia Gaze |
| No gradient training for this checkpoint | DISFA+, FacePlace, 300W, WFLW, AFLW, EYEDIAP |
| Excluded duplicate corpus | MPIIFaceGaze (overlaps MPIIGaze) |

Aff-Wild2's AU and V/A tracks are separate training pools from one dataset.
Each dataset's train/validation/test partitions must also be distinguished:
a held-out split does not mean the whole corpus was excluded from training.
The manuscript reports selection among six candidate checkpoints using
benchmark scores, including DISFA+ and EYEDIAP. These corpora are excluded
from gradient training but are not untouched by model selection.

The stage-1 and stage-2/3 training counts are 944,248 and 1,568,745 rows,
respectively. These counts and current-checkpoint roles come from the
manuscript's Tables 3 and S18. The inspected local training repository
confirms relevant source exclusions but does not contain a complete frozen
v2.8 run manifest. Earlier `_v2`, `_v26`, and `_v27` files require their own
inventory; this notice does not retroactively assign them the v2.8 split.

See [the dataset register](DATASETS.md) for individual source terms.
Validation-only use does not automatically relicense an otherwise
unmodified model. Data access, evaluation, model selection, and publication
still must satisfy the applicable agreement.

## Visualization models

The later `au_to_mesh`, `emotion_to_mesh`, and `bs_to_mesh` PLS artifacts
use CelebV-HQ frames and detector-generated targets; the v6 cards identify
Detectorv2 v2.8 as the teacher. Their fitted parameters have their own
provenance. Neither the software MIT license nor an assumption that all
teacher restrictions automatically transfer establishes their complete
distribution permissions. Existing valid grants are not revoked here.
