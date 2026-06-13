# Detectorv2 build plan (branch: feat/detector-v2-multitask)

Wire the v2.3 multitask model into py-feat as `Detectorv2`. Pipeline:
RetinaFace → v2.3 model (AU/emotion/gaze/landmark/pose/VA) + ArcFace identity → Fex.

## Decisions (from user)
- **Naming:** HARD rename. `Detectorv1` = old pipeline, `Detectorv2` = new. No bare `Detector`.
- **Output schema:** NATIVE v2 (24 AU, 8 emotion incl. Contempt, V/A, gaze, 6D pose, 478 mesh)
  PLUS a derived **dlib-68** landmark block from the 478 mesh via py-feat's
  `DLIB68_FROM_MP478` (feat/utils/blendshape_to_au.py) — so Fex helpers needing 68 points
  (plotting, PnP pose, alignment) keep working. Any function that needs 68 → feed it the
  converted 68.
- **HF:** push to `py-feat/face_multitask_v1` using token in /home/ljchang/Github/py-feat/.env
  (HF_TOKEN). ROTATE token after session.
- **Identity:** reuse existing ArcFace (r50) path unchanged.
- **Detection:** RetinaFace only.

## Preprocessing (MUST match training exactly — verified)
Training used py-feat's OWN `extract_face_from_bbox_torch`:
1. RetinaFace detect → bbox [x1,y1,x2,y2]
2. `extract_face_from_bbox_torch(frame_in_[0,1], bbox, face_size=256, expand_bbox=1.2)` → 256×256
3. SyncedAugment(train=False): resize 256→224, then ImageNet normalize
   (mean (0.485,0.456,0.406), std (0.229,0.224,0.225)). chip/255 first.
4. model(chip[B,3,224,224]) → dict

## Model load (from bench_gaze.py:60-84 pattern)
ckpt=torch.load(path,map_location,weights_only=False); cfg=ModelV2Config(**filtered_config,
pretrained=False); MEGraphAUv2(cfg).eval(); load_state_dict(ckpt["model"],strict=False).
Config in ckpt sets use_head_v3=True. Backbone convnextv2_tiny via timm (NEW dep).

## Output decode
- p_au [B,24] already prob[0,1] → 24 AU cols (AU_NAMES order)
- emotion_logits [B,8] → softmax → 8 emo cols (Neutral,Happy,Sad,Surprise,Fear,Disgust,Anger,Contempt)
- va [B,2] in [-1,1] → valence,arousal cols
- gaze [B,2] radians [yaw,pitch] → gaze cols (+ derived angle)
- pose [B,6] [yaw,pitch,roll,tx,ty,tz] radians/px
- mesh [B,478,3] chip-px coords → store 478 + derive dlib-68 block
- identity [B,512] from ArcFace

## Steps (dependency order)
1. [ ] ENV: add timm dep; resolve torchcodec import blocker for `import feat`.
2. [ ] feat/multitask/{model_v2.py,heads_v3.py} vendored (DONE). Add __init__.py.
3. [ ] feat/multitask/inference.py: MultitaskModel wrapper — load ckpt from HF, preprocess
       crop→chip, forward, decode to dict. Standalone unit test (no full Detector).
4. [ ] Repackage inference checkpoint (strip optimizer/EMA/train args; keep model+config) →
       smoke-test load+forward on feat/tests/data/single_face.jpg; sanity-check outputs.
5. [ ] Upload checkpoint + model card to py-feat/face_multitask_v1.
6. [ ] feat/detector.py: Detectorv2 class (RetinaFace → MultitaskModel → ArcFace → Fex
       native-v2 schema + dlib-68 derived). Loader pulls from HF into resources cache.
7. [ ] Hard rename Detector→Detectorv1 (update __init__ exports, internal refs). Keep tests.
8. [ ] Tests: feat/tests/test_detector_v2.py (single/multi face, batch consistency, Fex schema).
9. [ ] timm in pyproject.toml deps. Model card license notes (ArcFace non-commercial, ConvNeXt).

## Risks
- Preprocessing mismatch = silent accuracy drop. Mitigated: identical crop fn + verify.
- torchcodec blocking `import feat` is pre-existing env breakage, not ours.
- Hard rename breaks existing user code + all tests/docs referencing `Detector` — large blast radius.

## PROGRESS CHECKPOINT (session 2026-05-30)

VERIFIED WORKING (uv env):
- Dev env = uv: `cd /home/ljchang/Github/py-feat && uv run python ...`. The uv .venv has
  torch 2.11.0+cu128, feat 0.7.0 (editable -> THIS repo), timm 1.0.27. USE THIS.
  DO NOT use anaconda base (I broke it pip-installing torchcodec 0.13 incompatible w/ its
  old torch -> `register_fake` ImportError; `import feat` fails there). au_deep/.venv also
  works but uv is the project standard now.
- Branch: feat/detector-v2-multitask (off v0.7-dev).
- feat/multitask/{model_v2.py, heads_v3.py, __init__.py} vendored. FIX APPLIED: model_v2.py
  line 575 `from deep.heads_v3` -> `from feat.multitask.heads_v3` (the only deep.* import).
- Checkpoint loads: ModelV2Config(**filtered_config, pretrained=False) -> MEGraphAUv2 ->
  load_state_dict(ck["model"], strict=False) => ZERO missing/unexpected. Perfect match.
- Forward [B,3,224,224] -> dict (CONFIRMED shapes):
    p_au [B,24] (already prob 0-1), mesh [B,478,3], pose [B,6],
    emotion_logits [B,8], va [B,2] (valence,arousal tanh -1..1), gaze [B,2] (yaw,pitch rad),
    cooccur None.
  (Earlier "va [B,8]" note was WRONG — from a failed build before the import fix. va_head =
   nn.Linear(in_dim,2). Resolved.)
- ckpt keys: model, config, stage, epoch, metrics, args. Ship ck["model"].

NEXT: build feat/multitask/inference.py (preprocess crop->chip + forward + decode incl dlib-68
from 478 mesh via feat.utils.blendshape_to_au.DLIB68_FROM_MP478).
