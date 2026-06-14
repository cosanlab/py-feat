"""Verify every Hugging Face model artifact py-feat lazily downloads is live.

The detectors / plotting / eval paths pull weights from repos under the
``py-feat/`` org via ``hf_hub_download`` at first use. If a repo (or an
expected file in it) isn't published, users hit a 404 only when they first
exercise that code path — never at install time. This script resolves every
referenced repo up front (metadata only, no weight download) so a missing
artifact is caught before release.

The repo list is derived from grepping ``hf_hub_download(repo_id=...)`` /
``HF_REPO`` across ``feat/``. Keep it in sync if new models are wired in.

Usage:
    python scripts/verify_hf_models.py            # all repos
    python scripts/verify_hf_models.py --files    # also list each repo's files

Exits non-zero if any repo fails to resolve.
"""

import argparse
import sys

from huggingface_hub import HfApi
from huggingface_hub.utils import RepositoryNotFoundError, GatedRepoError

# (repo_id, [expected files] or None). Files checked only when given.
MODEL_REPOS = [
    ("py-feat/xgb_au", None),
    ("py-feat/svm_au", None),
    ("py-feat/svm_emo", None),
    ("py-feat/img2pose", None),
    ("py-feat/resmasknet", None),
    ("py-feat/facenet", None),
    ("py-feat/mobilefacenet", None),
    ("py-feat/mobilenet", None),
    ("py-feat/pfld", None),
    ("py-feat/l2cs", None),
    ("py-feat/pose_mlp_v2", None),
    ("py-feat/mp_blendshapes", None),
    ("py-feat/bs_to_au", None),
    ("py-feat/au_to_landmarks", None),
    ("py-feat/au_to_mesh", None),
    ("py-feat/landmarks68_to_mesh478", None),
    ("py-feat/retinaface_r34", None),
    ("py-feat/arcface_r50", None),
    ("py-feat/face_multitask_v2", None),  # Detectorv2
    ("py-feat/face_multitask_v1", None),
]

# Intentionally excluded: py-feat/mp_facemesh_v2 (the MediaPipe mesh model is
# mid-transition — legacy file slated for deletion — verify separately).
EXCLUDED = ["py-feat/mp_facemesh_v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", action="store_true", help="list files per repo")
    args = ap.parse_args()

    api = HfApi()
    failures = []
    print(f"Verifying {len(MODEL_REPOS)} HF model repos (excluding {EXCLUDED})\n")
    for repo_id, expected in MODEL_REPOS:
        try:
            info = api.repo_info(repo_id)
            files = [s.rfilename for s in info.siblings]
            missing = [f for f in (expected or []) if f not in files]
            if missing:
                failures.append((repo_id, f"missing files: {missing}"))
                status = f"FAIL  missing {missing}"
            else:
                status = f"ok    ({len(files)} files)"
            print(f"  {repo_id:<34} {status}")
            if args.files:
                for f in files:
                    print(f"       - {f}")
        except (RepositoryNotFoundError, GatedRepoError) as e:
            failures.append((repo_id, type(e).__name__))
            print(f"  {repo_id:<34} FAIL  {type(e).__name__}")
        except Exception as e:  # network, auth, etc.
            failures.append((repo_id, f"{type(e).__name__}: {e}"))
            print(f"  {repo_id:<34} ERROR {type(e).__name__}")

    print()
    if failures:
        print(f"{len(failures)} repo(s) failed to resolve:")
        for repo_id, why in failures:
            print(f"  - {repo_id}: {why}")
        sys.exit(1)
    print("All referenced HF model repos resolve.")


if __name__ == "__main__":
    main()
