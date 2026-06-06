# Cross-tool AU benchmark — methodology & integration notes

This documents how py-feat is compared against other open-source Python facial
action-unit (AU) toolkits, and — just as importantly — **what it took to run
each competitor**. The integration effort is itself a result: it shows the
usability gap between py-feat (`pip install py-feat`, one `Detector` /
`Detectorv2` call, runs on any GPU) and the alternatives.

## Feature comparison

Across the open-source Python facial-behavior toolkits. ✅ = supported,
❌ = not supported, ⚠️ = supported with a caveat (see notes).

| | **py-feat** | **OpenFace 3.0** | **LibreFace** | **PyAFAR** |
|---|:---:|:---:|:---:|:---:|
| **Install** | `pip install py-feat` ✅ | clone repo + checkpoints | `pip install libreface` ⚠️¹ | ❌ broken deps² |
| **Single images** | ✅ | ✅ | ✅ | ❌ **video only** |
| **Video** | ✅ | ✅ | ✅ | ✅ |
| **Action units** | ✅ 20 | ✅ 8 | ✅ 12 (+5 occ.) | ⚠️ 12 occ. / 5 int. |
| **AU intensity** | ✅ | ❌ occ. only | ✅ | ⚠️ 5 AUs |
| **Emotion** | ✅ 7-class | ✅ | ✅ | ❌ |
| **Valence / arousal** | ✅ (v2) | ❌ | ❌ | ❌ |
| **Gaze** | ✅ | ✅ | ✅ | ❌ |
| **Head pose (6DoF)** | ✅ | ✅ | ✅ | ❌ |
| **Landmarks** | ✅ 68 + 478 mesh | ✅ | ✅ 478 mesh | ❌ |
| **Identity / face ID** | ✅ ArcFace | ❌ | ❌ | ❌ |
| **One-call API** | ✅ `Detector().detect()` | ❌ custom scripts | ✅ ⚠️¹ | ⚠️ video only |
| **Latest GPUs (Blackwell)** | ✅ | ✅ | ❌ pinned old torch³ | ❌ dlib/CUDA³ |
| **License** | permissive⁴ | academic | USC research-only | non-commercial |

¹ The `pip` model is a distilled all-in-one net that **underperforms its own
  paper**; reproducing published AU numbers requires cloning the research repo +
  checkpoints (see LibreFace notes).
² Release wheel pins `pysimplegui==4.60.5`, which was pulled from PyPI — install
  fails; needs `--no-deps` + hand-resolving TF/MediaPipe/dlib + `download_models`.
³ LibreFace's pinned PyTorch lacks Blackwell (sm_120) kernels (≤ Ampere only);
  PyAFAR's dlib build fails compiling CUDA kernels.
⁴ py-feat is permissively licensed; a few downloadable weights (e.g. ArcFace
  identity) are research-only and clearly flagged.

**Takeaway:** py-feat is the only one of the four that installs with a single
`pip` command, takes both images and video, runs on current-generation GPUs, and
covers the full feature set (AUs + intensity, emotion, valence/arousal, gaze,
6DoF pose, 68/478 landmarks, identity) behind one API.

## Accuracy — AU detection on DISFA+ (held-out)

Mean per-AU **F1** on the DISFA+ benchmark (57,150 frames). **Protocols are not
yet fully harmonized** across tools (AU subset + binarization differ — see the
per-tool note); treat as indicative until a single-protocol recompute lands.

| Tool | DISFA+ mean F1 | AUs scored | binarization | source |
|------|:---:|:---:|---|---|
| **py-feat v2** (`Detectorv2`) | **0.540** | 12 | truth ≥2, prob ≥0.5 | `pyfeat_disfaplus_au.json` |
| **OpenFace 3.0** | 0.488 | 8 | their `evaluation.py` | `openface3_disfaplus.json` |
| **LibreFace** (research RepVGG) | 0.461 | 12 | truth ≥2, intensity ≥2 | `libreface_repvgg_disfaplus.json` |
| **py-feat v1** (`Detector`, xgb) | 0.250 | 12 | truth ≥2, prob ≥0.5 | `pyfeat_disfaplus_au.json` |
| **PyAFAR** | _n/a_ | ≤7 overlap | — | not runnable (see notes) |

**py-feat v2 (Detectorv2) leads** the held-out DISFA+ AU benchmark (0.54), ahead
of OpenFace 3.0 (0.49) and LibreFace (0.46) — and recall DISFA+ is held out for
*all* tools, while DISFA (LibreFace's/OF3's training set) is excluded. py-feat
v1's xgb path is weaker here (0.25) on the strict 12-AU / ≥2 protocol; it's the
legacy modular detector, and v2 is the recommended path.

LibreFace also gives mean intensity **PCC = 0.73** (its native DISFA metric).
A follow-up will recompute all tools on one AU set + threshold for an
apples-to-apples table.

## Accuracy — beyond AU: emotion, valence/arousal, gaze

AU is only one of the modalities these toolkits ship. We benchmark the rest on
the datasets that carry the right labels, each tool run **end-to-end as
written** (its own detector → its own model), on identical images/labels frozen
from py-feat's `feat.evaluation` loaders into shared manifests
(`shared/export_emotion_gaze_manifest.py`). Per-tool sample counts (`n`) differ
because each tool's *own* face detector decides which faces it finds — that
detection robustness is itself part of the comparison.

**Strictly out-of-sample.** Every number is on a held-out **validation/test**
split — no tool is ever scored on images from its own training set:

| Dataset | Split scored | Held out for |
|---|---|---|
| AffectNet | official **validation** (`validation_aligned.csv`) | all (tools train on AffectNet-train) |
| RAF-DB | **test** split | all (tools train on RAF-train) |
| DISFA+ | full posed-peak eval set | **all** — no tool trains on DISFA+ (we avoid DISFA, which several train on) |
| Columbia Gaze | full eval set | all |
| MPIIFaceGaze | held-out subsample | all |
| Gaze360 | **`bench`/test** split | all — incl. py-feat, whose L2CS trains only on Gaze360-train |

### Emotion — 7-class, top-1 accuracy / macro-F1

Held-out **AffectNet-val** (994 imgs, classes 0–6) and **RAF-DB test** (3,068).
All three emotion-capable tools argmax their own emotion head; OF3/LibreFace
emit 8 classes (incl. Contempt) — scored on the shared 7, a Contempt prediction
counts as wrong. (PyAFAR has no emotion head.)

| Tool | AffectNet acc / F1 | RAF-DB acc / F1 |
|------|:---:|:---:|
| **py-feat v2** | 0.492 / **0.479** | **0.656 / 0.528** |
| **OpenFace 3.0** | **0.493** / **0.520** | 0.513 / 0.469 |
| **LibreFace** | 0.455 / 0.403 | 0.646 / 0.386 |

py-feat v2 and OF3 are neck-and-neck on AffectNet (0.49); on RAF-DB py-feat
leads on both accuracy and the balanced macro-F1 (LibreFace's 0.646 accuracy but
0.386 macro-F1 is the majority-class/Happiness skew).

### Valence / arousal — CCC (AffectNet-val)

**py-feat v2 is the only tool of the four that predicts continuous valence and
arousal at all** — so this isn't a head-to-head, it's a capability the others
lack. On AffectNet-val: **valence CCC 0.535, arousal CCC 0.482**.

### Gaze — mean angular error across three datasets

Each tool emits gaze in its own (yaw, pitch) frame, sign, and unit, mostly
undocumented. We resolve that I/O convention **identically and in every tool's
favor** — the single best discrete (axis × sign × unit) mapping to the GT frame
(`shared/gaze_convention.py`) — so no tool is penalised for an opaque output
convention. The same procedure is applied to py-feat.

| Tool | Columbia (1.2k) | MPIIFaceGaze (3k) | Gaze360 test (2.5k) |
|------|:---:|:---:|:---:|
| **py-feat v2** | **2.72°** | **2.80°** | **9.42°** |
| **OpenFace 3.0** | 12.05° | 7.03° | 41.09° |
| **LibreFace** | 15.40° | 19.49° | 32.08° |

py-feat's L2CS gaze leads on all three, and is strikingly stable (~2.7–2.8°) on
the two in-the-wild/lab sets where the competitors swing widely. Two caveats,
both stated rather than hidden:

- **Columbia loader fix.** The Columbia yaw-sign convention was corrected in
  `feat.evaluation` as part of this work (it had reported 17.5° from a sign
  mismatch, not a model error); py-feat's stock harness now reports 2.72°.
- **Gaze360 is in-distribution for py-feat.** Its L2CS model trains on
  Gaze360-*train*; we score the held-out `bench`/test split (no train leakage),
  but the *distribution* is still home-field for py-feat and out-of-distribution
  for OF3/LibreFace — so read Gaze360 as "how far OOD pushes each tool" (OF3
  collapses to 41° on the ±170° poses), not a like-for-like ranking. Columbia
  and MPIIFaceGaze are held out for everyone equally.

> Reproduce: `tools/<tool>/run_accuracy.py` (OF3/PyAFAR), `run_modalities.py`
> (LibreFace), `run_gaze.py` + `run_pyfeat_modalities.py` (py-feat). Consolidated
> by `ingest_accuracy.py` into `accuracy.csv`; published to the
> `py-feat/benchmarks` HF dataset.

## Speed

Throughput on the **shared test fixtures** (`single_face.mp4` video + a
`multi_face.jpg` image batch — *not* the accuracy datasets), each tool timed
end-to-end (detect → AU), across a hardware × batch matrix:

**Hardware:** CPU · RTX 3090 (sm_86) · RTX PRO 6000 Blackwell (sm_120) · Apple M5 (MPS)
**Batch:** 1 (single frame) and 16

**A blank cell is data.** If a tool can't run on a given device it gets *no
number* — that absence documents the tool's hardware reach. Expected coverage:

| Tool | CPU | 3090 | Blackwell | M5 (MPS) |
|------|:---:|:---:|:---:|:---:|
| **py-feat** | ✅ | ✅ | ✅ | ✅ |
| **OpenFace 3.0** | ✅ | ✅ | ? | ? |
| **LibreFace** | ✅ | ✅ | ❌ (no sm_120) | ? (cuda/cpu API) |
| **PyAFAR** | ✅? | ? | ❌ (dlib/CUDA) | ❌ (Ubuntu/WSL2 only) |

py-feat's own CPU/3090/Blackwell numbers are in the **[live dashboard](live.md)**
(e.g. Detectorv2 ≈ 285 fps on Blackwell batch 16); M5 is added from a Mac run.

**Methodology** (this matters — naive timing is misleading): every tool is timed
**end-to-end** (decode → detect → AU, the full pipeline it ships), on the **same
video** (`WolfgangLanger_Pexels.mp4`, 472 frames), with **warmup + 3 repeats**
(median reported) and `torch.cuda.synchronize()` around GPU work. Crucially, the
head-to-head is at **batch 1** — OpenFace 3.0 and LibreFace process per-frame
(their APIs don't expose batching), so comparing them to py-feat's batched
throughput would be apples-to-oranges.

**Head-to-head — RTX 3090, end-to-end, batch 1:**

| Tool | fps | vs py-feat |
|---|:---:|:---:|
| **py-feat Detectorv2** | **38.4** | — |
| **OpenFace 3.0** | 18.3 | 2.1× slower |
| **LibreFace** | 4.7 | 8.2× slower |
| **PyAFAR** | n/a | — |

**py-feat's batching is a separate advantage:** its `detect()` natively batches,
so Detectorv2 scales **38 → 202 fps** from batch 1 to 16 on the 3090. OF3 and
LibreFace have no batch path in their APIs (their *models* can batch, but the
shipped pipeline doesn't), so they stay at the per-frame rate. LibreFace's GPU
barely helps it at all — its MediaPipe alignment is CPU-bound and dominates.

The CPU / Blackwell / M5 cells need the same rigorous harness (folded into the
suite's `run_speed.py`); the earlier single-clip per-frame numbers there are
**not** trustworthy and were withdrawn. Hardware reach is still data: **OF3 runs
on Blackwell; LibreFace and PyAFAR cannot** (no sm_120 / dlib-CUDA).

The point of the matrix is exactly the blanks: py-feat is the only toolkit that
runs across CPU, current-gen GPUs, *and* Apple Silicon — and is one-to-two orders
of magnitude faster where competitors do run.

## Datasets & metric protocol — and why **DISFA+**, not DISFA

The cross-tool comparison runs on **DISFA+** (posed-peak, 12-AU intensity), the
held-out benchmark py-feat reports against (Cheong et al. 2023) and the dataset
our existing OpenFace 3.0 result already uses (`"dataset": "disfaplus"`).

**We deliberately do *not* evaluate on DISFA.** DISFA is the **training set** for
LibreFace (and is used by OpenFace 3.0), so scoring those tools on DISFA is
in-distribution — a home-field advantage and effective train/test contamination.
DISFA+ is held out for all tools, so it measures **generalization**: a tool that
only does well on its own training distribution is exactly what a fair benchmark
should expose. (We verified the in-distribution case as a *sanity check* — the
LibreFace RepVGG model tracks AU intensity cleanly on DISFA — then evaluate the
real comparison on DISFA+.)

Metrics: per-AU **PCC** (intensity, threshold-free — LibreFace's native metric)
and binary **F1** at intensity **≥ 2** on both prediction and ground truth
(matching `feat.evaluation.metrics`' truth convention). Where a tool emits
probabilities (py-feat, OF3) rather than intensities, its native binarization is
noted per table so protocols are never silently mixed.

## Per-tool integration experience

### py-feat (v1 / v2)
`pip install py-feat`; one call returns AUs (+ emotion, pose, gaze, landmarks,
identity). Runs on CPU, CUDA (incl. **Blackwell / RTX PRO 6000**, sm_120, on
torch 2.11+cu128), and Apple MPS. No per-tool preprocessing to match.

### LibreFace — could **not** reproduce published numbers locally
A multi-day saga that is worth recording in full:

1. **The pip API (`libreface.get_facial_attributes`) collapses on this data.**
   Its `au_intensities` are near the noise floor for clearly-active AUs — e.g.
   a DISFA+ frame labeled AU12 intensity 4 (a posed smile) returns
   `au_12 ≈ 0.015` (it correctly returns ~2.5 on a normal smiling photo).
   No threshold/normalization recovers a signal that isn't there.
2. The LibreFace **paper** reports DISFA AU *intensity* via **PCC** (0.63) from a
   **separate research module** (`AU_Recognition`, RepVGG checkpoint), not the
   distilled all-in-one pip model. So we cloned the repo and loaded
   `new_checkpoints_fm_repvgg/DISFA/all/repvgg.pt` (output `×5 → [0,5]`).
3. **The research checkpoint can't run on Blackwell.** LibreFace pins an old
   PyTorch built for **sm_37…sm_86**; Blackwell is **sm_120**. Weights copy to
   the GPU ("loaded"), then the first compute kernel aborts (no sm_120 binary).
   **LibreFace is therefore restricted to ≤ Ampere GPUs** — we benchmark it on
   the RTX 3090 (sm_86). py-feat runs on Blackwell unchanged.
4. **It is fragile to out-of-distribution data and alignment.** On **DISFA+**
   (posed-peak) the research checkpoint *also* collapsed (AU12 ramp 0→4 stayed
   ~0.1–0.5, non-monotonic) — under both LibreFace's own MediaPipe alignment and
   DISFA+ native `Aligned/` crops. It only works on **DISFA itself**, fed
   **DISFA's own aligned crops** (`DISFA_/aligned/...`): there the AU12 ramp is
   clean and monotonic — GT 0→0.05, 1→~0.5, 2→~1.2, 3→~2.7, 4→~3.5, 5→~4.4.

**Status:** the research RepVGG checkpoint runs correctly. On the held-out
**DISFA+** set (its own aligned crops, LibreFace's test transform, 57,150 frames)
it generalizes reasonably: **mean F1 = 0.46, mean PCC = 0.73** over 12 AUs
(`run_libreface_repvgg_disfaplus.py` → `libreface_repvgg_disfaplus.json`).
Strong on AU25 (F1 0.91), AU04/09 (0.75); weak on AU06 (0.11), AU15/17/20
(~0.06). Note: earlier single-frame probes suggested a "collapse" — that was an
artifact of unrepresentative frames and the broken *pip* model; the full-dataset
research-model run is the truth, and it's fine. (The lesson — over-concluding
from a handful of frames — is why we report the whole-benchmark output.)

Getting even this far required: discovering the pip model is a different
(weaker) net than the paper's, cloning the research repo + checkpoints, working
around a Blackwell-incompatible torch (3090-only), and matching the alignment —
vs. py-feat's `pip install` + one call. The silent failure modes (pip-model
collapse, out-of-distribution collapse) are themselves the usability story:
without careful per-frame validation you'd ship wrong numbers.

### OpenFace 3.0
Cleaner to install than LibreFace/PyAFAR (`pip install openface-test` +
`openface download`), but the shipped pip CLI has **three blocking bugs**:
1. A **hardcoded developer path** baked into the STAR landmark config
   (`ckpt_dir = '/work/jiewenh/openFace/OpenFace-3.0/STAR'`) → `Permission
   denied: '/work'` on any other machine until patched.
2. `openface detect-video` throws an **OpenCV `imread` error** (mishandles video
   frames) — video mode is unusable.
3. `openface ... -d cuda` always raises *"provide at least one valid device
   ID"* — the CLI never passes `device_ids`, so **GPU mode is broken** from the
   CLI.

Critically, **the CLI also hardcodes `device='cpu'`** inside `process_image`
(line 23) — so `openface detect -d cuda` silently runs on CPU. The fix is to
**bypass the CLI** and construct the pipeline objects directly with real device
handling: `FaceDetector(device='cuda')`, `LandmarkDetector(device='cuda',
device_ids=[0])`, `MultitaskPredictor(device='cuda')`. Done that way OF3 runs
fine on GPU **including Blackwell** (modern torch).

Once the CLI is bypassed, OF3 is solid: AU accuracy on DISFA+ **0.488** (8 AUs)
and end-to-end speed **18.3 fps** on the 3090 (batch 1, rigorous harness —
warmup + repeats, median). So OF3 is the *least* broken competitor — it just
needs the CLI worked around. (The running, once set up, is reliable — the
friction is all in packaging.) Note its **MTL AU model batches fine** (the model
takes `[B,…]`; the shipped pipeline just feeds one face at a time), so OF3 *could*
be sped up with a custom batched runner — its API simply doesn't.

### PyAFAR — dependency rot + API/coverage mismatch
Another multi-obstacle integration (its own MediaPipe + TensorFlow env, kept
away from the torch stack):

1. **Won't `pip install`.** The release wheel pins `pysimplegui==4.60.5`, a GUI
   library that PySimpleGUI **pulled from PyPI** (2024 licensing change), so the
   dependency is unsatisfiable — a GUI pin blocks a headless benchmark. Workaround:
   install the wheel `--no-deps` and hand-resolve the real runtime deps
   (`tqdm`, `scipy`, `h5py`, `tensorflow`, `mediapipe`, `opencv`, …) one ImportError
   at a time.
2. **dlib won't build.** `pip install dlib` compiles from source and fails on its
   **CUDA** kernels (same Blackwell sm_120 wall); the documented path is a **conda**
   env with conda-forge's precompiled dlib.
3. **Video-only API.** `adult_afar(filename=<video>, …)` takes a video file —
   DISFA+ is per-frame stills, so frames must be re-assembled into per-trial
   videos to feed it.
4. **Partial AU coverage.** Its occurrence AUs {1,2,4,6,7,10,12,14,15,17,23,24}
   overlap DISFA's 12 on only **7** (1,2,4,6,12,15,17); intensity on **3**
   (6,12,17). It cannot score the full DISFA AU set.
5. Non-commercial license; GPU only on Ubuntu/WSL2.

Contrast: py-feat is one `pip install`, works headless, takes images or video,
and reports all 20 AUs. [PyAFAR DISFA+ numbers — pending the conda env + a
frame-to-video adapter; coverage limited to the 7 overlapping AUs.]

## Hardware notes

- **LibreFace can only be benchmarked on ≤ Ampere GPUs (e.g. RTX 3090, sm_86)**
  because its pinned PyTorch lacks Blackwell (sm_120) kernels. All LibreFace
  numbers here are 3090 runs.
- py-feat and the competitor runs that use modern torch run on Blackwell
  (RTX PRO 6000) and the 3090 alike.

## License caveats (research benchmark only)

- LibreFace: USC research-only. PyAFAR: non-commercial. OpenFace 3.0: check its
  license. py-feat's competitors are **never** added as py-feat dependencies —
  each runs in its own isolated env (`~/benchmark-envs/<tool>`), consuming a
  frozen DISFA manifest (`scripts/competitors/`) so the comparison shares
  identical frames/labels without coupling envs.

## Reproducing

- Manifest (py-feat env): `scripts/competitors/export_disfaplus_manifest.py`
  (DISFA equivalent forthcoming).
- LibreFace research model: clone `ihp-lab/LibreFace`, use `AU_Recognition`
  RepVGG checkpoint, **on a 3090**.
- py-feat v1/v2 accuracy: `scripts/bench_accuracy.py --detector v1|v2`.
