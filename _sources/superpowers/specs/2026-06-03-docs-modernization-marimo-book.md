# Documentation modernization: marimo-book migration, live benchmarks, pyfeat-live

**Status:** Planning / research (no implementation in progress)
**Author:** Luke Chang (with Claude Code)
**Date:** 2026-06-03
**Target release:** **v2.0** — the major release that ships everything on `v0.7-dev`
plus current work. The docs relaunch is part of the v2.0 cut, not a separate effort.
**Related:** current docs build (`docs/_config.yml`, `docs/_toc.yml`,
`.github/workflows/manual_docs.yml`), `scripts/bench_detectors.py`

## Context

py-feat docs are currently built with **Jupyter Book 1** (Sphinx-based `jb-book`):
12 `.ipynb` tutorials + 26 MyST `.md` pages, notebooks **not** executed at build
(`execute_notebooks: off`, outputs pre-baked), API reference via Sphinx
`autodoc` + `napoleon`, deployed to **py-feat.org** via gh-pages.

We want to modernize for v2.0. The maintainer of py-feat also maintains
**marimo-book**, so adopting it here is *dogfooding*: py-feat is a hard,
realistic workout (torch-heavy notebooks, academic citations, multi-tutorial
structure, an API reference) and its unmet needs become marimo-book's feature
backlog. This inverts the usual "is this tool mature enough?" risk — we control
the tool, and notebooks-as-pure-Python + Markdown means content is portable
regardless.

**Browser-execution reality (researched, settled):** py-feat cannot run in
Pyodide/WASM — `torch`, `torchvision`, `torchcodec`, `xgboost`, `onnxruntime`
have no Pyodide wheels, and the WASM tier has a 2 GB cap and no threading.
PyTorch-in-Pyodide is not on any realistic horizon. **This is not a blocker:**
marimo-book's static tier renders pre-computed outputs at build time (matching
today's `execute: off` behavior), and **molab** ("Open in molab" buttons)
provides real cloud execution — torch pre-installed, 4 CPU / 32 GB RAM,
optional Blackwell GPU. So: static render on-page, live GPU execution on demand.
A future in-browser demo would be a separate ONNX + onnxruntime-web project, not
"py-feat in the browser."

## Goals

1. Relaunch py-feat.org on **marimo-book** for v2.0, modern and maintainable.
2. Tutorials authored as marimo `.py` notebooks (reactive, git-friendly,
   `molab`-runnable); prune internal/validation notebooks off the public site.
3. A **live benchmarks** page: periodically-updated throughput + accuracy trends.
4. A **pyfeat-live** docs section with auto-integrated changelogs and a
   download-latest-release button per supported platform.
5. An **API reference** (approach deferred — see Workstream E).
6. Use py-feat's needs to drive a prioritized **marimo-book feature backlog**.

## Workstreams

| ID | Workstream | Status |
|----|------------|--------|
| A | jupyter-book → marimo-book migration | **Spike done (2026-06-03)** — green w/ 2 findings |
| B | Tutorial restructure (prune + port + new) | Plan ready |
| C | Live benchmarks page + data pipeline | Architecture chosen; one open decision |
| D | pyfeat-live docs (changelog + downloads) | Pattern chosen |
| E | API reference | **Deferred — needs research** |
| F | GitHub org + repo structure | Open decisions |
| G | marimo-book feature backlog (dogfooding) | Backlog drafted |
| H | Blog / news / announcements | Plan ready (→ marimo-book module) |

---

## Workstream A — marimo-book migration

**Why marimo-book over Jupyter Book 2:** JB2 keeps `.ipynb`/MyST but is a
different goal (we want to modernize the notebook format, not preserve it) and
*also* lacks autodoc, so it wins us nothing on the one hard piece. marimo-book
gives reactive pure-Python notebooks, `molab` live execution, and a tool we
control.

**Conversion is built-in:** `marimo convert nb.ipynb -o nb.py` (outputs stripped
on convert). The real per-notebook labor is the **reactive-DAG refactor** —
marimo forbids redefining the same variable across cells, so tutorials that
incrementally mutate a `fex` dataframe across cells need restructuring.

### Phases

1. **Spike (de-risk first).** On a `docs-marimo` branch: convert
   `docs/basic_tutorials/01_basics.ipynb`, stand up a 2-chapter marimo-book
   (one converted notebook + one `.md` page). Confirm: (a) static tier renders
   pre-computed torch outputs without trying to execute, (b) "Open in molab"
   actually runs py-feat, (c) which Markdown features survive (admonitions,
   cross-refs, the raw `<iframe>` in `intro.md:36`). The gaps found here feed
   Workstream G.
2. **Config & TOC.** Translate `_config.yml` + `_toc.yml` → marimo-book
   `book.yml`. Re-create logo/favicon, repo buttons, Google Analytics,
   launch buttons.
3. **Port core notebooks** (Workstream B) with DAG refactor.
4. **Migrate prose `.md` pages**, fix Markdown-flavor gaps, handle the iframe.
5. **CI/deploy.** Replace `jupyter-book build docs` in `manual_docs.yml` (and
   the auto-deploy workflow) with the marimo-book build; keep the `py-feat.org`
   CNAME on gh-pages.
6. **Cut over** for v2.0; drop JB1 deps from `requirements-dev.txt`.

### Risks

- **Markdown feature parity** (admonitions/cross-refs/citations) — see spike: math,
  raw HTML/iframe, and autoref cross-refs all work; **MyST admonitions do not**.
- **Reactive refactor** effort per notebook — mechanical but real.
- **Versioned docs** — py-feat ships breaking releases; pinning docs to a release
  is a need (→ Workstream G item 5).

### Spike results (2026-06-03) — RESOLVED

Built a throwaway 2-chapter book (converted `01_basics` + a feature-probe `.md`
page + a torch demo notebook) with marimo-book 0.1.18 / marimo 0.23.8. Findings:

- ✅ **Conversion works.** `marimo convert 01_basics.ipynb` → clean 49-cell `.py`,
  no convert-time errors. (Reactive-DAG conflicts — redefining a var across cells —
  surface only at *run* time, so that per-notebook refactor effort is still real.)
- ✅ **Torch outputs render statically** without a kernel: tensor reprs, DataFrames
  (→ HTML `<table>`), and matplotlib figures (→ base64 PNG) all rendered.
- ✅ **Launch buttons work** and are repo-aware (molab / GitHub / Download all
  emitted, pointing at the configured repo/branch).
- ✅ **Math (MathJax), raw-HTML `<iframe>`, and autoref cross-references** (`[Text][]`
  → `href="demo/#anchor"`) all work — so `intro.md`'s iframe and equations survive.
- ❌ **MyST admonitions do NOT render** — `` ```{note} `` / `:::{note}` leak as
  literal text. Material syntax (`!!! note`) is required. **Scope: 11 occurrences
  across 5 pages** (`installation`, `models`, `au_reference`, `contribute`,
  `modelContribution`) — mechanical conversion.
- ⚠️ **Build executes notebooks** (`marimo export ipynb --include-outputs`), unlike
  today's `execute_notebooks: off`. So docs CI needs `feat`+torch+models (and a GPU
  for acceptable speed); content-hash caching re-runs only changed chapters. → new
  marimo-book feature candidate (G item 9).
- ⚠️ **MkDocs 2.0 churn** — Material for MkDocs prints a MkDocs-2.0 "no migration
  path / plugins removed" warning. Forward-looking risk for the toolchain to track.

**Verdict:** migration is green. The only content blocker is the 11-admonition
rewrite; the only infra change is build-time notebook execution (CI on a GPU runner
or a no-execute mode — G item 9).

---

## Workstream B — tutorial restructure

The "advanced" track is mostly internal model-validation notebooks, not user
docs. Prune them off the public site.

| Notebook | Action |
|----------|--------|
| `01_basics`, `02_detector_vids`, `03_plotting`, `04_fex_analysis` | **Port to marimo** (core public tutorials) |
| `09_test_bbox`, `10_test_lands`, `11_test_Poseinfo`, `12_test_aus`, `13_test_emos` | **Drop from site** → move to `examples/` or fold into `feat/tests/` |
| `06_trainAUvisModel`, `07_extract_labels_and_landmarks`, `08_train_hogs` | **Keep one** reframed as "Contributing a model" (intro courts CV researchers); drop the other two or rely on `modelContribution.md` |

**New tutorials** (author natively as marimo notebooks; fill real gaps):

1. **Detectorv2 walkthrough** — the headline v2.0 detector, currently undocumented:
   when to use vs classic `Detector`, 20-AU/7-emotion outputs, accuracy/speed.
2. **Identity & face tracking** — ArcFace embeddings, clustering across video,
   honest limitations (v0.8 clustering-rewrite roadmap).
3. **Statistical analysis of `Fex`** — regression, t-tests, ISC, predicting
   conditions from AUs (the analysis layer that distinguishes py-feat).
4. **Performance & devices** — `num_workers=0`, batching, MPS/CUDA, large-dataset
   processing (encodes the CLAUDE.md gotchas so they stop being FAQ tickets).

---

## Workstream C — live benchmarks

### Core principle: results-as-data (decouple compute / storage / presentation)

`scripts/bench_detectors.py` already captures rich provenance per run (py-feat
version, git sha, host + CPU arch, Python, PyTorch build, GPU model/CUDA, OMP
threads, full config sweep) and emits markdown. **First step is a `--json`
emitter** that serializes those records (metadata block + one object per
`device × batch × workers × config` measurement). That single change is the
stable interface; everything downstream consumes it and the storage sink is
*pluggable and reversible*.

### Compute — split by benchmark type (different invariants)

- **Throughput lane → cron on the maintainer's server (fixed hardware).** fps is
  only comparable on the same machine over time. Plain `cron` running
  `bench_detectors.py --json`, tagged by host/GPU. Trigger weekly + on release tag.
  *Avoid wiring the server in as a self-hosted GitHub Actions runner:* GitHub
  prohibits self-hosted runners on public repos (fork PRs run arbitrary code).
  Plain cron pushing results sidesteps this.
- **Accuracy lane → anywhere with a GPU; HF Scheduled Jobs is a clean fit.**
  Hardware-independent. **HF Scheduled Jobs** = "cron jobs on GPU on the Hub"
  (`huggingface_hub` v0.35+), pulls the **private** eval datasets (DISFA+,
  AffectNet, RAF-DB, AffWild2 — licensed → private), emits F1 records. Or just
  run it in the same server cron to start.

**Correctness rule for plots:** segment by (host, GPU, config) — never draw fps
across different machines as one line.

### Storage / data management — options researched

| Option | Reactivity | Versioned | Backend / creds | Verdict |
|--------|-----------|-----------|------------------|---------|
| **Append JSON/CSV to versioned store** (HF Dataset or `benchmark-results` repo) | Rebuild-triggered ("daily-fresh") | Yes (git/HF) | None for reads; token for writes | **Recommended default.** Simplest, provenance-rich, public reads need no auth. HF Dataset adds a free viewer. |
| **Firebase Firestore** (cron writes via Admin SDK; docs read client-side) | **True real-time** (onSnapshot, no rebuild) | No (point-in-time) | GCP project + security rules; JS SDK in docs | **Recommended upgrade path.** Free tier ample (50k reads/day); read-only public via rules. Benchmark *plotting* needs no torch, so it runs in marimo-book's browser tier / an anywidget / a vanilla-JS island. This is the "reactive docs" idea, done right. |
| **Google Sheets** (cron appends; read via gviz CSV/JSON) | Near-live (client fetch) | No | "anyone with link"; gviz endpoint | **Prototype / throwaway only.** Trivial to append and human-editable, but not a real time-series store; fine for a phase-0 spike, not the canonical store for a flagship project. |

**Recommendation:** start with **append-JSON to a versioned store** (HF Dataset)
+ build-time static render in marimo-book, rebuilt on schedule / via
`repository_dispatch` when new data lands. Because the `--json` interface is
fixed, **swapping to Firebase later for true real-time is a sink change, not a
recompute** — so we are not locked in. Keep results **public**; keep the licensed
eval *data* **private** (separate stores). Do **not** commit the high-frequency
time-series into the main repo git history (bloat); keep curated milestone
reports in `docs/benchmarks/*.md` as today.

> **Open decision (C-1):** ship v2.0 with the static-fresh default, or invest in
> the Firebase real-time path up front? Leaning static-fresh first; Firebase as
> fast-follow (and a marimo-book "live data widget" feature — Workstream G item 3).

### Presentation

A marimo dashboard chapter that loads the results store and renders trends:
fps per (host, GPU, config) over releases; F1 per (model, dataset) over releases.
Static-rendered at build; "Open in molab" for a live re-run against latest data.

### Optional: regression alerting

`benchmark-action/github-action-benchmark` ingests the `--json`, maintains the
series, renders trend charts, and **comments on PRs when a metric regresses**.
Runs on GitHub-hosted runners (no GPU, no self-hosted-runner issue). Add if we
want "py-feat got slower" caught automatically. (Not `asv` — it rebuilds a
per-commit env, fighting heavyweight GPU model runs.)

### Shape

```
server cron ──(throughput .json)──┐
                                   ├──► py-feat/benchmarks ──► marimo dashboard
HF Scheduled Job ─(accuracy .json)─┘    (HF Dataset, public)    (static + molab)
   └─ pulls private eval datasets             │
                                              └──► github-action-benchmark (optional)
   [later] swap sink → Firestore for real-time client-side updates
```

---

## Workstream D — pyfeat-live docs section

Reuse the **pattern** from docs.hyperstudy.io (Docusaurus: versioned "Release
Notes" section), but implement it in marimo-book for site consistency, and
**automate** the changelog (hyperstudy's version pages appear hand-maintained).

GitHub Releases on the `pyfeat-live` repo are the source of truth.

1. **Overview / getting started** — it's a downloadable app, not `pip install`;
   needs its own onboarding flow distinct from the library.
2. **Per-platform install guides** (macOS / Windows / Linux — confirm supported set).
3. **Downloads page** —
   - *Primary CTA:* client-side **OS-detecting "Download for your platform"**
     button hitting `api.github.com/repos/<org>/pyfeat-live/releases/latest`,
     auto-selecting the matching asset (one-script-tag pattern, no deps).
   - *Full list:* build-time-templated per-platform table + version string from
     the Releases API (works offline / SEO).
4. **Changelog / release notes** — auto-generated Markdown pages from GitHub
   Releases (one per version + a "latest"), rebuilt via `repository_dispatch`
   when pyfeat-live publishes a release. No hand-maintained changelog pages.

This whole pattern ("release-notes + downloads from GitHub Releases") is a strong
candidate marimo-book built-in (Workstream G item 4) — Docusaurus-plugin-equivalent.

---

## Workstream E — API reference (DEFERRED — needs research)

The one hard gap. marimo-book has no docstring/autodoc support (neither does
MyST-MD, so it's not a reason to prefer JB2). py-feat uses NumPy/Google-style
docstrings (currently rendered via Sphinx napoleon).

**Options to evaluate (not yet decided):**

- **Sidecar `pdoc`** → standalone static API site, near-zero config, reads
  existing docstrings; mount at `py-feat.org/api`, link from the book.
  Pragmatic stopgap; ships v2.0 without blocking on tooling.
- **Build autodoc *into* marimo-book** — a docstring directive (likely via
  `griffe`, the engine behind mkdocstrings) emitting Markdown pages the book
  ingests. Every Python-library doc site wants this; py-feat's docstrings are a
  good test corpus. Higher effort; a marimo-book feature, not just a py-feat fix.
- **Generate Markdown from docstrings externally** (`pydoc-markdown`/`griffe`)
  and feed as plain Markdown pages — middle ground.

**Open question (E-1):** sidecar `pdoc` for v2.0 + built-in autodoc as a
fast-follow marimo-book release? (Maintainer wants more research before deciding.)

---

## Workstream F — GitHub org + repo structure (open decisions)

Creating a **`py-feat` GitHub org** (matches the `py-feat/` HuggingFace org and
the py-feat.org domain) makes sense for housing `py-feat`, `pyfeat-live`, the
benchmarks dataset, and possibly the docs/site.

**Open decisions:**

- **F-1 — Transfer `py-feat` to the org now or after v2.0 ships?** Transfer sets
  up redirects (clone URLs, the pinned `@v0.7.0` install URL should survive via
  redirect — *verify*), but Zenodo DOI webhook, Coveralls, and badges need
  re-pointing. Recommendation: create the org now, start *new* repos there
  immediately (pyfeat-live, benchmarks); transfer the flagship as a separate
  checklisted step.
- **F-2 — Docs in-repo vs standalone portal repo?** In-repo keeps executable
  tutorials + autodoc next to code (versions together). Standalone makes sense
  only if py-feat.org becomes a multi-product portal (py-feat + pyfeat-live).
  Given pyfeat-live is now in scope, a portal is plausible — but the cost
  (cross-repo aggregation, version pinning) is real. Decide alongside F-1.

---

## Workstream G — marimo-book feature backlog (py-feat-driven dogfooding)

Priority order for a scientific Python package:

1. **API reference from docstrings** (see E) — biggest gap; every library wants it.
2. **Scientific-Markdown features** — py-feat has an arXiv ID + Zenodo DOI;
   needs **citations/BibTeX**, cross-references, admonitions, math. Unlocks the
   whole computational-science docs audience, not just py-feat.
3. **"Live data dashboard" pattern** — the benchmarks page: a marimo notebook
   that reads a remote store (HF Dataset / Firestore / HTTP), renders trends,
   static-renders at build, "Open in molab" for live re-run. Templatize it →
   flagship example + reusable feature.
4. **Releases pattern** — auto changelog + OS-detecting downloads from GitHub
   Releases (Workstream D), as a built-in.
5. **Versioned docs** — pin a build to a release; serve multiple versions.
6. **Heavy-dependency static degradation** — confirm torch notebooks render via
   precomputed outputs and the build never tries to execute them.
7. **Blog/news module** (Workstream H) — dated posts, tags, authors, RSS/Atom.
8. **Generic SSG plumbing** — search, sitemap, RSS, OpenGraph/SEO, analytics,
   **redirects** (needed for JB1→marimo URL migration), 404.
9. **No-execute / cached-output mode** (from spike; **design locked 2026-06-04**) —
   heavy GPU notebooks shouldn't re-execute on every docs deploy.
   - *Why not marimo's own cache:* `mo.persistent_cache` is explicitly
     non-portable (docs recommend gitignoring `__marimo__/cache/`), keys aren't
     machine-independent, and return values must pickle — so it can't ship
     pre-rendered outputs to a clean CI runner. It stays useful for the *local*
     re-render loop, not CI.
   - *Mechanism:* the outputs already exist in committable form — marimo-book's
     **rendered `.md`** (post-`marimo export`, outputs embedded as HTML/base64).
     Reuse the existing content-hash export cache, but persist the rendered output
     as a **committed `_rendered/` dir** keyed by source hash + tool version +
     `book.yml` signature. Author renders once locally (GPU); CI reuses, never
     executes — i.e. jupyter-book `execute: off` semantics.
   - *Opt-in:* per-chapter `mode: cached` in the TOC (mirrors existing
     `mode: wasm`), so heavy notebooks use committed outputs while light ones
     execute fresh.
   - *Staleness UX:* warn when a chapter's committed output hash ≠ current source;
     `marimo-book render` regenerates, `marimo-book render --check` is a CI gate.
   - **STATUS: implemented (2026-06-04)** on `marimo-book` branch
     `feat/cached-render-mode`. New `mode: cached` (book-default + per-entry),
     `RenderedStore` (committed `_rendered/` + manifest keyed by source hash),
     `marimo-book render [--check]` CLI, and build-time reuse with live-render
     fallback. 8 new tests + full suite (198) green; ruff clean; end-to-end CLI
     verified that `build` reuses committed output with **zero** notebook
     execution. Only the notebook *body* is committed, so config/TOC changes
     don't invalidate it.
10. **MyST-admonition compatibility (optional)** — accept `` ```{note} ``/`:::{note}`
    and map to Material admonitions, easing migrations from jupyter-book/MyST.

---

## Workstream H — blog / news / announcements

A blog/news section for releases, announcements, and updates. This is canonical
SSG functionality (Docusaurus and most doc tools ship it built-in), so it should
be a **marimo-book module, not a py-feat one-off** (see split below).

A blog module wants: a `blog/` of dated Markdown (or marimo `.py`) posts with
frontmatter (title, date, authors, tags); a paginated index; tag/author pages;
per-post permalinks; an **RSS/Atom feed**; and a "latest posts" widget for the
landing page. py-feat's first posts: the v2.0 launch, the Detectorv2
announcement, the new benchmarks page.

It also composes with Workstream D: a pyfeat-live release can auto-create a blog
post from its GitHub Release notes.

---

## Build into marimo-book vs py-feat one-off — the split

Direct answer to "what belongs in marimo-book rather than a custom one-off here."
**Principle:** the *generic mechanism* goes in marimo-book; the *project-specific
config/content* stays in py-feat. Benchmark since hyperstudy is on Docusaurus —
treat Docusaurus's built-ins as the bar for "a docs SSG should just have this."

| Feature | Where | Why | Prior art |
|---------|-------|-----|-----------|
| **Blog / news** (Workstream H) | **marimo-book module** | Every docs site wants news; generic posts+tags+RSS | Docusaurus blog plugin |
| **API reference from docstrings** (E) | **marimo-book module** | Every Python-library site wants autodoc; py-feat docstrings are the test corpus | mkdocstrings/`griffe`, Sphinx autodoc |
| **Scientific Markdown** — citations/BibTeX, cross-refs, admonitions, math (G2) | **marimo-book module** | Unlocks the whole research-docs audience, not just py-feat | MyST-MD, Quarto |
| **Releases: auto-changelog + OS-detect download buttons** (D) | **marimo-book module** | Many projects ship apps/binaries from GitHub Releases | Docusaurus release plugins |
| **Live-data dashboard widget** — read remote store, render trends, static+molab (G3) | **marimo-book module/template** (reusable widget) | Reusable across any data/benchmark site | Observable, Evidence.dev |
| **Versioned docs** (G5) | **marimo-book module** | Common library need (multiple released versions) | Docusaurus/Sphinx versioning |
| **Generic plumbing** — search, sitemap, RSS, OpenGraph/SEO tags, analytics injection, redirects, 404 | **marimo-book core** | Table-stakes SSG features; needed for the JB1→marimo URL migration (redirects) | All mature SSGs |
| Benchmark **schema + accuracy harness** (configs, AU/emotion F1, eval-data pulls) | **py-feat one-off** | Domain-specific to py-feat's models/datasets | — |
| Tutorial **content**, org/repo wiring, py-feat `book.yml` config | **py-feat one-off** | Project-specific by nature | — |

So almost everything we'd be tempted to hand-roll here (blog, downloads, autodoc,
citations, live dashboard, versioning) is better as a marimo-book module — py-feat
becomes the driving use case for each. The genuinely py-feat-specific pieces are
only the benchmark *content/harness* and project config.

---

## Decisions: locked vs open

**Locked:**
- Adopt marimo-book for v2.0 (not Jupyter Book 2).
- Prune internal `test_*` notebooks; port the 4 core tutorials; add 4 new ones.
- Live benchmarks via results-as-data; `--json` emitter is the stable interface.
- Default storage = append-JSON to a versioned store (HF Dataset), reversible to
  Firebase. Results public, eval data private.
- pyfeat-live section with automated GitHub-Releases changelog + downloads.
- Blog/news section; build it as a marimo-book module, not a py-feat one-off.
- Generalizable features (blog, downloads, autodoc, citations, live dashboard,
  versioning) → marimo-book modules; only benchmark schema/harness + content +
  project config stay py-feat-specific.

**Open:**
- C-1: static-fresh vs Firebase real-time for v2.0.
- E-1: API reference approach (sidecar `pdoc` vs built-in autodoc) — needs research.
- F-1: when to transfer `py-feat` to the new org.
- F-2: docs in-repo vs standalone portal repo.

## Suggested sequencing toward v2.0

1. `--json` emitter for `bench_detectors.py` (unblocks all of C; tiny).
2. marimo-book spike (A phase 1) — validates migration + generates G tickets.
3. Create `py-feat` org + `pyfeat-live` repo (F, new repos only).
4. Port core tutorials + migrate prose pages (A/B).
5. Benchmarks page (C) — static-fresh default.
6. pyfeat-live docs section (D) + blog/news module (H).
7. API reference (E) — sidecar `pdoc` to ship v2.0.
8. CI/deploy cutover (with redirects from old JB1 URLs); drop JB1.

## References

- marimo-book — https://marimobook.org/
- marimo: WASM limits — https://docs.marimo.io/guides/wasm/ ; convert —
  https://docs.marimo.io/guides/coming_from/jupyter/ ; molab —
  https://docs.marimo.io/guides/molab/
- Pyodide package constraints — https://pyodide.org/en/stable/usage/wasm-constraints.html
- HF Scheduled Jobs — https://huggingface.co/docs/hub/jobs-schedule
- HF leaderboard data guide — https://huggingface.co/docs/hub/leaderboard-data-guide
- github-action-benchmark — https://github.com/benchmark-action/github-action-benchmark
- Firestore pricing/free tier — https://firebase.google.com/docs/firestore/pricing
- Google Sheets gviz — https://hooshmand.net/google-sheets-json-database/
- pdoc — https://pdoc.dev/ ; griffe — https://mkdocstrings.github.io/griffe/
