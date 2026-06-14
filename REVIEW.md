# Review & triage guidelines for Claude Code automation

These instructions steer the Claude Code GitHub Action (PR review, issue
triage, and `@claude` mentions). They complement `CLAUDE.md` — read that too;
its conventions and load-bearing gotchas always apply.

## PR review

**Lead with what matters. Keep nits few.** A review that buries one real bug
under ten style nits is worse than a short one. Order findings by severity and
say plainly when the PR looks good.

Flag as **Important**:
- Correctness / logic bugs, especially in the multi-face / batched paths
  (`(image|video) × (single|multi-face) × (batch=1|batch>1)` is the matrix
  that has historically broken — see `feat/tests`).
- Breaking public-API changes without a deprecation path.
- New public function/keyword that's documented but unused in the body, or a
  parameter that silently does nothing (this has happened — e.g. an animation
  `frame_duration` that wasn't wired up). Either wire it or don't ship it.
- Device-handoff bugs: tensors created on CPU while weights are on MPS/CUDA.
- Missing test coverage for new behavior (the suite targets the device/shape
  matrix; new code paths should be exercised).
- `try/except` scaffolding that masks a symptom instead of fixing root cause.

**Do NOT flag** (these are deliberate — see `CLAUDE.md`):
- The `OMP_NUM_THREADS=1` block in `feat/__init__.py`.
- The `_patch_xgboost_setstate_for_skops` shim.
- The xgb/svm `*_v2.skops`→v1 and `mp_facemesh_v2` TorchScript→legacy fallbacks.
- `num_workers=0` as the default.
- Absence of AI attribution in commits — that's required, not an oversight.

**Style:** match `CLAUDE.md` — comments only where the WHY is non-obvious; no
multi-paragraph docstrings on internal methods. Don't ask for comment churn.

The review is advisory. Do not formally approve or request-changes; a human
maintainer runs the merge.

## Issue triage

- **Require a reproduction for bugs.** If steps are missing, add `needs-repro`
  and ask once for: py-feat version, OS/Python, a minimal code snippet, and
  expected vs. actual. Don't interrogate; one friendly ask.
- **Check before treating as new.** Many v0.6-era requests are already handled
  in v0.7 (HuggingFace model hosting, disabling sub-models via
  `Detectorv1(identity_model=None)`, batched HOG, etc.). Cross-check `CLAUDE.md`
  and recent closed issues; if resolved, say so and point to where.
- **Roadmap awareness:** identity/tracking and real-time work is v0.8
  (`docs/superpowers/specs/2026-05-02-identity-detection-roadmap.md`);
  benchmarks and docs migration are in flight. Label such requests as roadmap
  items rather than asking for more detail.
- **Labels:** use only labels that already exist (`gh label list`); never
  invent new ones. At most 2–3 per issue. Don't close issues.

## Cost / scope

Stay focused and bounded — these run on a finite monthly credit pool. Answer
the question asked; don't expand scope or open large investigations unless a
maintainer explicitly asks (`@claude` with a specific request).
