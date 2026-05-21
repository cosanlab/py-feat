# Identity detection roadmap

**Status:** Vision / planning (no implementation in progress)
**Author:** Luke Chang (with Claude Code)
**Date:** 2026-05-02
**Related PRs:** #284 (ArcFace as default identity model)

## Context

Identity detection in py-feat has two layers: the **embedding model** that turns face crops into vectors, and the **clustering** that groups vectors into named identities. PR #284 upgraded the embedding model from FaceNet (triplet loss, VGGFace2) to ArcFace R50 (angular margin softmax, WebFace600K) — a clear quality lift, validated on `multi_face.jpg` where ArcFace's max cross-similarity (0.35) gives comfortable margin while FaceNet's (0.76) was misclustering at typical thresholds.

The clustering layer is unchanged from v0.6: cosine-similarity > threshold → connected components → `Person_<k>` labels. This is the "light and flexible but not particularly intuitive or good" piece. The embedding upgrade improves raw signal but doesn't fix the clustering problems.

## Goals

A v0.8+ identity workflow that:

1. **Doesn't require a magic threshold to be right.** Today users tune `face_identity_threshold` per-video and it doesn't generalize.
2. **Can say "I don't know."** Today every face gets a label, even faces too blurry/occluded to be reliably identified — those should be `Unknown`.
3. **Supports cross-video continuity.** A longitudinal-study user processing video 1 then video 2 should be able to track that "subject A" in both is the same person.
4. **Supports user-supplied references.** When users have known reference images for known subjects, identities should come back as `Alice` / `Bob`, not `Person_0` / `Person_1`.
5. **Returns confidence, not just labels.** Downstream analysis often wants to weight or filter by identification confidence.
6. **Is incremental where possible.** Adding new video to an existing dataset shouldn't require re-clustering everything from scratch.

## Concrete problems with the current approach

1. **Brittle hard threshold.** A single `threshold=0.8` needs to work across single-subject vlogs, group conversations, and noisy crowd footage. It can't.
2. **Connected-components is too transitive.** One above-threshold edge between A↔B and B↔C merges A↔C even when A and C are unambiguously different people.
3. **No quality awareness.** Low-confidence detections (occlusion, motion blur, partial faces) embed and cluster alongside high-quality frontal faces.
4. **No "unknown" output.** Forces every face into some cluster.
5. **Order-dependent labels.** `Person_0`, `Person_1`, ... numbering depends on encounter order. Re-running the same `detect()` with different `batch_size` can swap labels.
6. **No cross-video persistence.** Each `detect()` invocation produces an independent label set.
7. **No user gallery.** Users with known reference subjects can't anchor clustering to those identities.

## Approach options

| Approach | What it solves | Cost / risk |
|---|---|---|
| **HDBSCAN** with cosine metric | brittle threshold, transitivity (density-adaptive) | new dep (~5MB), ~10× compute over current |
| **Quality filtering** | "Unknown" output, fewer false-positive merges | tiny — 1 boolean per face |
| **Gallery / reference-face mode** | stable named identities, cross-video continuity, user knowledge | modest — accept dict of name→image, embed each, match |
| **Tracker-based per-video continuity** (DeepSORT-lite) | same person across frames stays one ID without relying on embedding similarity | heavy — new dep, integration with `detect_faces` loop |
| **`IdentityConfidence` column** | downstream filtering / reweighting | small — compute centroid post-hoc |
| **Hierarchical clustering with dendrogram output** | visualizable; user can re-cut without rerunning embeddings | modest — `scipy.cluster.hierarchy`, save linkage matrix |
| **Online/incremental clustering** | append-only growth without recomputation | medium — BIRCH or online agglomerative; order-sensitive |
| **Quality-aware embedding** (MagFace) | quality is a side-output of the embedding itself | requires re-embedding model; same retrieval as ArcFace otherwise |

## Three layered designs to consider

### Design A: Drop-in replacement (smallest scope, ~150 LOC)

Switch the default clustering algorithm from connected-components to HDBSCAN, add quality filtering, add an `IdentityConfidence` column. Keep the same API surface.

```python
fex = detector.detect(video, identity_threshold=0.5, min_face_score=0.9)
fex.Identity            # 'Person_0', 'Person_1', 'Unknown'
fex.IdentityConfidence  # 0.0-1.0, cosine to cluster centroid
```

- **Pros:** Better defaults, no API churn, preserves backwards compatibility.
- **Cons:** Doesn't address cross-video continuity or known-subject workflows.

### Design B: Gallery-aware (medium scope, ~250 LOC)

Adds an explicit way for users to provide reference faces:

```python
detector = Detector(
    identity_gallery={
        "Alice": "alice_ref.jpg",
        "Bob": ["bob_1.jpg", "bob_2.jpg"],  # multiple refs averaged
    }
)
fex = detector.detect(video)
fex.Identity  # 'Alice', 'Bob', 'Unknown_0', 'Unknown_1' (auto-discovered for the rest)
```

Logic: if cosine similarity to any gallery prototype exceeds threshold, label with the prototype's name. Otherwise fall back to discovered clustering for the unknowns.

- **Pros:** Most intuitive for FEAT's longitudinal-study use case. Falls back to Design A when no gallery is provided.
- **Cons:** Some users won't have reference images; doesn't help in pure exploratory mode.

### Design C: Track-then-cluster (heaviest, ~500 LOC)

Within a video, use a face tracker (simple IoU+embedding tracker; full DeepSORT is overkill) to maintain identity across nearby frames. Each track gets a single embedding (averaged over the track's frames). Then cluster *tracks* rather than individual face detections. Within a track, identity is stable by construction.

- **Pros:** Fixes the hardest case (same person across many frames with varied pose/expression); much cleaner per-frame output.
- **Cons:** Adds tracking machinery to `detect_faces`; more state to manage.

## Suggested staging

| PR | Scope | Why this order |
|---|---|---|
| 0 | **Labeled-video benchmark harness** — load a clip with per-frame ground-truth identity, compute cluster F1 / V-measure / ARI for any chosen identity pipeline | Without it we're tuning by feel. Should arguably come first to validate every later step against ground truth. |
| 1 | Design A: HDBSCAN + quality filter + `IdentityConfidence` | Immediate visible quality lift, small surface area, no new user concepts. |
| 2 | Design B: gallery-aware mode | Solves the cross-video / known-subjects problem most longitudinal users actually have. |
| 3 | Design C: track-then-cluster | Highest per-frame quality but biggest scope. Defer until 1+2 are validated. |

## Open questions

1. **Benchmark data:** Does cosanlab have a labeled clip from prior studies, or do we need to annotate one? Cheap option: a podcast/interview clip (2-5 known speakers, 10-30 min) with manual per-frame identity labels.
2. **HDBSCAN as a dep:** Comfortable with adding it? Well-maintained, ~5MB install, but isn't currently pulled by py-feat.
3. **Default behavior with no gallery:** When users don't specify `identity_gallery`, should we auto-discover (current behavior) or refuse and require explicit method choice (`method='discover'`)?
4. **Confidence semantics:** Is "cosine to cluster centroid" the right confidence definition, or should we compute something more like "margin to second-nearest cluster"? Margin is more useful for downstream filtering but more compute.
5. **Cross-video state:** Should the gallery be a `Detector` constructor arg (immutable) or a `Detector.add_subject(name, image)` method (mutable, builds up over a study)? The mutable version supports the "add as the study progresses" pattern but complicates reproducibility.

## Out of scope for v0.7

This roadmap is for v0.8+. v0.7 ships:

- ArcFace as default identity model (PR #284)
- Existing cluster_identities (with the `cluster_identities` perf rewrite from PR #283)

Everything in this doc is a follow-up.
