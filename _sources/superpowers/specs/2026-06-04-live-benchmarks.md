# Live benchmarks — implementation spec

**Status:** Design locked; ready to implement
**Author:** Luke Chang (with Claude Code)
**Date:** 2026-06-04
**Parent:** [docs-modernization spec](2026-06-03-docs-modernization-marimo-book.md) (Workstream C)
**Depends on:** `scripts/bench_detectors.py --json` (done, on `v0.7-dev`),
`feat/evaluation/` accuracy harness (exists; needs a `--json` emitter)

## Decisions

**Settled:**
- **Repo split (py-feat must not depend on OpenFace/LibreFace):** py-feat keeps its
  *own* benchmark tooling (no competitor deps) and produces the **"all py-feat
  features"** report. A **separate `benchmark-suite` repo** (future `py-feat` org)
  holds the **cross-tool comparison** — one isolated-env adapter per tool; OF3 and
  LibreFace are dependencies of *that* repo only. Adding a tool = adding an adapter.
  Two data products: `pyfeat_full.csv` (py-feat's own tooling) and `cross_tool.csv`
  (suite, shared metrics only).
- **Store:** a public **HF Dataset** `py-feat/benchmarks`.
- **Liveness:** `mode: cached` dashboard page, **rebuilt on new data** (not
  real-time). The page reflects the latest run, refreshed automatically — without
  a Python kernel in the page or a Firebase backend.
- **Format: CSV** (not parquet). Reasons: trivial read-modify-write append with
  pandas, the HF Dataset Viewer renders it, it's human-diffable, and it keeps the
  door open to a future `mode: wasm` "live-on-load" upgrade (Pyodide pandas reads
  CSV without pyarrow). Parquet buys nothing at this data volume (KB/week).

**Defaults (adjustable):**
- **Build order:** throughput lane first (its data already flows via `--json`),
  proving store→dashboard end-to-end before wiring accuracy.
- **Accuracy compute:** HF Scheduled Jobs (keeps the licensed eval datasets on the
  Hub) — a fast-follow after throughput.
- **Cadence:** weekly + on every release tag.

**Why not the alternatives** (see parent spec): a git results repo is the equal
runner-up (loses the free Viewer + data API); Firebase only adds real-time *push*,
which weekly data doesn't need; static-rebuild gives "live enough" without the
Pyodide tax on every visitor.

## Architecture

```
liquidswords2 cron ──bench_detectors.py --json──┐
  (+ Blackwell/GB10, tagged by host)            │  ingest: flatten JSON → append CSV
                                                ├─► py-feat/benchmarks (HF Dataset)
HF Scheduled Job ──accuracy --json──────────────┘     throughput.csv
  └─ pulls PRIVATE eval datasets (token)              accuracy.csv
                                                          │
                              repository_dispatch  ◄──────┘ (on append)
                                                          │
                              marimo-book build ──► dashboard page (mode: cached,
                                                    Open-in-molab; future mode: wasm)
                                                          │
                              github-action-benchmark ◄───┘ (optional regression PR comments)
```

## Data

**Two CSVs** in the dataset (different schemas, both denormalized — one row per
measurement, run metadata repeated per row, so a dataframe/plot needs no join):

`throughput.csv`
```
date, tool, tool_version, git_commit, host, machine, gpu, python, pytorch, omp_num_threads,
section_kind, section_label, config, device, batch, workers, sec, n_units, fps, per_unit_ms
```

`accuracy.csv`
```
date, tool, tool_version, git_commit, host, gpu, python, pytorch,
dataset, split, held_out, metric_kind, metric_name, value, n
```
`held_out` = was this dataset absent from the tool's training set (per its paper);
`split` = which split was scored (e.g. `val`/`test`). The cross-tool head-to-head
filters to `held_out` True for all compared tools.

The **`tool`** dimension (`py-feat-v1`, `py-feat-v2`, `openface3`, `libreface`) is
what makes this a cross-tool comparison; `config` stays for py-feat sub-configs
(face/au/emotion model choices). Every tool's runner emits the *same*
`{metadata, records[]}` JSON, so the store/ingest/dashboard are tool-agnostic.

Both derive 1:1 from the `{schema_version, metadata, records[]}` JSON the emitters
produce — the **ingest step flattens** `metadata` across each `records[]` row.

**Append mechanism (single weekly writer → no race):** an ingest script
downloads the current CSV via `huggingface_hub`, appends the new rows, and uploads
a new commit. `schema_version` lives in the JSON; a column-set change bumps it and
the ingest script migrates.

## Compute

- **Throughput (fixed hardware):** cron runs `bench_detectors.py --json`. Tag each
  run by `host`/`gpu` (already in the metadata) so liquidswords2, Blackwell, and
  GB10 are distinct series. **Done:** the emitter. **To build:** the cron unit +
  the ingest step.
- **Accuracy (hardware-independent, needs private data):** add a `--json` emitter
  wrapping `feat.evaluation.evaluate_all_datasets` (same `{metadata, records[]}`
  shape; one record per (detector, dataset, metric)). Run it as an HF Scheduled
  Job that pulls the private eval datasets with a token.

## Cross-tool comparison (py-feat v1/v2 + OpenFace 3 + LibreFace)

The benchmark compares **four tools**: py-feat `Detector` (v1), py-feat
`Detectorv2`, **OpenFace 3.0**, and **LibreFace**. Two problems the store doesn't
solve:

**1. Environment isolation — and where the code lives.** py-feat must not depend on
OpenFace 3 / LibreFace, so the competitor adapters live in the **separate
`benchmark-suite` repo** (future `py-feat` org), each in its own isolated env,
each emitting the common `--json`:
- `adapters/pyfeat/` → reuses py-feat's *own* `bench_detectors.py --json` + accuracy
  emitter (pip-installed py-feat; no duplicated logic).
- `adapters/openface3/` → isolated venv (CMU OpenFace 3.0); `openface3_deps/` is
  empty — needs setup.
- `adapters/libreface/` → the `libreface` PyPI package.
py-feat's **full-feature** report runs straight from py-feat (no suite, no
competitor deps). The adapters are the only tool-specific code; ingest/store/
dashboard never change. Natural fit for **per-tool uv venvs** (same host → fair
throughput) or per-tool HF Job images.

**2. Metric alignment — the credibility crux.** Tools predict different AU sets and
emotion taxonomies, so naive comparison is invalid. The protocol:
- **AUs:** report per-AU F1 only on the **intersection** of (each tool's AUs) ∩
  (the dataset's labeled AUs). py-feat outputs 20 AUs; OpenFace 3 ≈ the classic
  17 (`feat.utils.openface_AU_list`: 1,2,4,5,6,7,9,10,12,14,15,17,20,23,25,26,45);
  LibreFace ≈ 12. The comparable set is the **shared AUs**, listed explicitly on
  the page; per-tool extras are shown separately, never in the head-to-head.
- **Emotions:** compare only on shared classes (all four do the basic 7; contempt
  and valence/arousal are tool-specific add-ons reported separately).
- **Throughput:** end-to-end `detect()` on **identical inputs**, **same hardware**,
  one warmup + one timed call — already how `bench_detectors.py` works; the OF3 /
  LibreFace runners mirror it. Pipelines differ (each does its own
  face→landmark→AU/emotion), so the number is "wall-time to full output," stated
  as such.

**3. Train/test contamination — the validity gate.** Most standard AU/emotion
benchmarks are *training data* for these tools, so naive comparison is invalid:

Per-task eligibility (train = in that tool's training set; — = not used; ✅ = clean):

*Action Units*
| Dataset | OF3 | LibreFace | py-feat | head-to-head |
|---|---|---|---|---|
| DISFA / DISFA+ | train | train | train (v1) | ❌ |
| BP4D | train | train (CV) | train (v1) | ❌ |
| CK+ / UNBC / AFF-Wild2 | — | — | train (v1) | ❌ (py-feat) |
| EmotioNet | — | pre-train | ? | ⚠️ |
→ **No AU set is clean for all three.** A fair AU head-to-head needs a dataset
none used (e.g. GFT, SAMM, CASME) or a freshly-labeled set.

*Emotion*
| Dataset | OF3 | LibreFace | py-feat | head-to-head |
|---|---|---|---|---|
| AffectNet | train | train | likely (resmasknet) | ❌ |
| RAF-DB | — | train | ? (v2 reports it) | ⚠️ LibreFace train |
| FER+/ExpW/CK+/JAFFE | — | — | some (v1 svm) | ⚠️ |
→ RAF-DB is clean for **OF3 only**; need a set none trained on.

*Gaze*
| Dataset | OF3 | py-feat (L2CS) | head-to-head |
|---|---|---|---|
| MPII Gaze / Gaze360 | train | train | ❌ |
| **Columbia Gaze** | — | eval only | ✅ **clean for both** |
→ **Gaze has a clean head-to-head: Columbia Gaze** (neither trained on it;
LibreFace has no gaze). `load_columbia_gaze` already exists.

Sources: OpenFace 3.0 = [arXiv 2506.02891](https://arxiv.org/abs/2506.02891)
(AU: DISFA+BP4D; gaze: MPII Gaze+Gaze360; emotion: AffectNet-8); LibreFace =
[arXiv 2308.10713](https://arxiv.org/abs/2308.10713) (AU: pre-train
EmotioNet/AffectNet/FFHQ → DISFA+BP4D; emotion: AffectNet+RAF-DB); py-feat =
`docs/pages/models.md`. **The obvious benchmarks (DISFA+/BP4D/AffectNet) are out
for the fair head-to-head** — every tool has seen them. **Gap to close:** py-feat
`face_multitask_v2`'s exact training sets (and `resmasknet`'s) — needed to finish
the AU/emotion columns; the maintainer has these from the v2 training repo.

Rules:
- **Per-(tool, dataset) eligibility.** Add a **`held_out` boolean** to each accuracy
  record (per tool × dataset), sourced from each tool's paper. The cross-tool view
  plots only cells `held_out=True` for *every* compared tool; `pyfeat_full.csv` may
  still show contaminated benchmarks but flagged `held_out=False` ("trained-on").
- **Validation/test splits only** — never score a tool on its train split
  (`load_affectnet_val` already does this).
- A truly-held-out set for all three likely needs a less-common or newer dataset
  (or a fresh labeled set); candidates need per-paper confirmation.
- **TODO before finalizing:** read both competitor PDFs end-to-end to lock the
  exact per-tool training-set lists → the eligibility matrix.

**Provenance:** each tool's `--json` carries its own `tool_version` (py-feat
version, OpenFace 3 release, libreface version) so a tool upgrade is a visible
trend break, not silent.

## Store — `py-feat/benchmarks` (HF Dataset, public)

- Public reads (dashboard fetches without auth). A **write token** (HF) lives in
  the cron/Job secret store.
- **Results are public; the eval *datasets* stay private** — separate repos.
- The Dataset Viewer gives a free browsable table immediately, before the
  dashboard exists.

## Trigger / freshness

1. cron/Job produces `run.json`.
2. ingest appends rows to the relevant CSV and commits to the Dataset.
3. that commit fires a **`repository_dispatch`** to the docs repo (a tiny step in
   the same cron, using a GitHub token).
4. the docs workflow runs `marimo-book build`; the dashboard chapter reads the CSV
   at build and re-renders. Page is fresh within minutes of each run.

## Dashboard page (`docs/benchmarks/` chapter, `mode: cached`)

A marimo notebook that reads both CSVs from the Dataset and renders trends:

- **Throughput:** fps over releases (x = `feat_version`/date), **one line per
  `(host, gpu, config)`** — never collapse across hardware. Faceted by device/batch.
- **Accuracy:** F1/CCC over releases, one line per `(detector, dataset, metric)`.
- A "latest run" summary table at the top (most recent per host/config).
- **"Open in molab"** lets a reader re-run the dashboard against the very latest
  data on a real kernel.

**Correctness rules (bake into the notebook):**
- Throughput is only comparable within a fixed `(host, gpu, config)`. Group on
  these columns; if a hardware tag is missing, drop the row rather than mis-plot.
- Plot from the CSV's own columns; tolerate `schema_version` drift (filter to the
  current column set, log dropped older-schema rows — no silent truncation).

**Future upgrade (documented, not built):** flip the chapter to `mode: wasm` to
fetch the CSV in-browser on each visit (live-on-load, no rebuild) — works because
the store is CSV. No Firebase required.

## Optional — regression alerts

`benchmark-action/github-action-benchmark` consumes the same `--json` and comments
on a PR when fps/F1 regresses past a threshold. Runs on GitHub-hosted runners (no
GPU), so it's independent of the compute lanes.

## Security / hygiene

- HF write token + GitHub dispatch token in cron/Job secrets only.
- **No self-hosted GitHub Actions runner on the public repo** (fork PRs run
  arbitrary code) — plain cron pushes results instead.
- Curated milestone reports stay in `docs/benchmarks/*.md` (hand-reviewed); the
  high-frequency time-series lives only in the Dataset, never committed to the
  main repo's history.

## Implementation steps (phased)

**Phase 1 — throughput end-to-end (proves the loop):**
1. Create the public `py-feat/benchmarks` HF Dataset with an empty `throughput.csv`
   (header only) + a README documenting the schema.
2. `scripts/ingest_benchmarks.py`: read a `--json` file → flatten → append to the
   Dataset CSV via `huggingface_hub` (read-modify-write) → commit.
3. A cron unit on liquidswords2: `bench_detectors.py --json` → `ingest_benchmarks.py`
   → fire `repository_dispatch`.
4. The dashboard chapter (marimo `.py`, `mode: cached`) reading `throughput.csv`;
   add it to `book.yml`. Render once, commit `_rendered/`.
5. Docs workflow: handle the `repository_dispatch` event → `marimo-book build`.

**Phase 2 — accuracy lane:**
6. Accuracy `--json` emitter over `evaluate_all_datasets`.
7. HF Scheduled Job: pull private datasets → run → `accuracy.csv` ingest.
8. Extend the dashboard with the accuracy panels.

**Phase 3 — external tools (the cross-tool comparison):**
9. Decide the shared-AU / shared-emotion comparison sets (intersection across
   py-feat, OpenFace 3, LibreFace, and each dataset's labels).
10. Scaffold the **`benchmark-suite` repo** with `adapters/pyfeat/` (reusing
    py-feat's `--json`) and the orchestration that appends `cross_tool.csv`.
11. `adapters/openface3/` — isolated env (populate `openface3_deps/`), wrap OF3's
    detect into the common `--json`; `adapters/libreface/` via the `libreface`
    PyPI package. Run all tools on the **same host** for fair throughput.
12. Add `tool`/`tool_version` to every record (py-feat emitter maps config→tool;
    suite adapters set it directly). Extend the dashboard with a cross-tool page
    plotting per-tool lines on the shared metrics, tool-specific outputs separate.

**Phase 4 — polish:**
13. `github-action-benchmark` regression alerts (optional).
14. Document the `mode: wasm` live-on-load upgrade path.

## Open / deferred
- Exact throughput sweep to schedule (all 4 configs × which hosts) and accuracy
  dataset set — start with the current `bench_detectors.py` defaults + the
  `feat/evaluation` default datasets; trim if runs get long.
- Whether to also run accuracy on liquidswords2 instead of HF Jobs (ops simplicity
  vs keeping licensed data on the Hub).
- **Cross-tool decisions:** the exact shared-AU set and shared-emotion set; the
  env strategy for OF3/LibreFace (per-tool uv venvs vs per-tool HF Job images);
  which datasets the external tools are run on; and who installs/configures
  OpenFace 3 (`openface3_deps/` is empty) and LibreFace. Throughput comparison
  needs all tools on the *same* host to be fair.

## References
- HF Scheduled Jobs — https://huggingface.co/docs/hub/jobs-schedule
- HF Dataset Viewer / data API — https://huggingface.co/docs/hub/datasets-viewer
- github-action-benchmark — https://github.com/benchmark-action/github-action-benchmark
- `bench_detectors.py --json` — `v0.7-dev` `2832e0a`
