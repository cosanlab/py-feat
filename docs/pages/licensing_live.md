<!-- Mirrored from LICENSES/PYFEAT-LIVE.md; keep the notices synchronized. -->

# PyFeat-Live licensing

Reviewed October 8, 2026.

## Application code

The PyFeat-Live application source code is licensed under its
[MIT License](https://github.com/cosanlab/pyfeat-live/blob/main/LICENSE),
including commercial use. Third-party libraries retain their own licenses.

## Downloaded models and optional features

The application downloads and runs Py-Feat models. Its MIT license does
not replace the terms for those pretrained weights or their source data:

- Classic detector configurations: [Py-Feat v1.0](licensing_v1.md).
- Multitask detector configurations: [Py-Feat v2.0](licensing_v2.md).
- Identity, gaze, pose, and visualization assets must be checked separately
  for the chosen model and checkpoint. Disabling ArcFace alone does not
  make the default multitask model commercially licensed.
- Optional generator assets have their own model cards and upstream
  dependencies. This review does not certify those separate generative
  checkpoints for unrestricted use.

The application wrapper neither removes nor adds to a provider's rights
in the models. The [dataset register](licensing_datasets.md) records training and
evaluation provenance and unresolved permissions. It does not require
every application user to sign every dataset agreement merely to run
inference; any such access requirement must come from the applicable model
or provider terms.

This notice does not grant dataset access, revoke existing valid licenses,
or establish a blanket license over users' inference outputs.
