"""Fit the v2.4 AU->mesh Procrustes-PLS deformation model.

Same recipe as fit_pls_variants.py (per-frame Umeyama similarity Procrustes +
iterative GPA to a population mean, pose-filter, PLSRegression scale=True), but:
  - 20-AU v2.4 input (drops AU16/18/27/45)
  - TARGET = the v2.4 model's OWN 478-mesh (captured per chip) — so AU->mesh
    deformation is consistent with what Detectorv2 actually outputs ("our mesh").
Source = the fast chip-based extraction (v24chips_celebvhq_{a,b}_*).

  uv run python scripts/fit_pls_v24_mesh.py --out-dir /Storage/Projects/mp_blendshapes/data/pls_v24
"""
import argparse, math, os
import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression

CHIPS = "/Storage/Projects/mp_blendshapes/data"
AU20 = ["AU01","AU02","AU04","AU05","AU06","AU07","AU09","AU10","AU11","AU12",
        "AU14","AU15","AU17","AU20","AU23","AU24","AU25","AU26","AU28","AU43"]
POSE = ["Pitch", "Yaw", "Roll"]
N_VERT = 478
MESH_ANCHORS = [10, 9, 8, 151, 6, 168, 197, 195, 33, 263, 133, 362]
YAW_LIMIT, PITCH_LIMIT = math.radians(40.0), math.radians(30.0)
GPA_MAX_ITER, GPA_TOL, K = 5, 1e-3, 40
CONDITIONS = ("standard", "nopose", "noid")
# Landmarks for the canonical frontal basis (midline + outer eye corners)
I_FOREHEAD, I_CHIN, I_EYE_R, I_EYE_L = 10, 152, 33, 263


def _unit(v):
    return v / (np.linalg.norm(v) + 1e-12)


CANONICAL_FACE = "/home/ljchang/Github/py-feat/feat/resources/canonical_face_model.pt"
I_CHEEK_R, I_CHEEK_L = 234, 454
_CANON = None


def canonical_face():
    """The MediaPipe canonical neutral face (468,3) — the natural frontal pose."""
    global _CANON
    if _CANON is None:
        import torch
        c = torch.load(CANONICAL_FACE, weights_only=False)
        C = np.asarray(c["vertices"] if isinstance(c, dict) and "vertices" in c else c, float)
        if C.ndim == 2 and C.shape[0] == 3 and C.shape[1] != 3:
            C = C.T
        _CANON = C
    return _CANON


def align_to_canonical(M):
    """Rotation-only Procrustes (Kabsch) aligning the neutral template to the
    canonical MediaPipe face over the shared 468 vertices. new = old @ R.T.
    Pins yaw/roll AND pitch to the canonical's natural frontal pose — more robust
    than a 3-landmark basis, whose forehead->chin chord isn't anatomically
    vertical (forehead sits behind the chin in depth) and tilts the face up."""
    C = canonical_face()
    P = M[:C.shape[0]]
    Pc = P - P.mean(0); Qc = C - C.mean(0)
    Pc = Pc / np.linalg.norm(Pc); Qc = Qc / np.linalg.norm(Qc)
    U, S, Vt = np.linalg.svd(Pc.T @ Qc)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


def _hw(M):
    return abs(M[I_FOREHEAD, 1] - M[I_CHIN, 1]) / abs(M[I_CHEEK_R, 0] - M[I_CHEEK_L, 0])


def aspect_scale_y(mean_frontal):
    """Factor to scale mesh-y so its height/width matches the canonical MediaPipe
    face (un-warps the square-chip horizontal stretch)."""
    return _hw(canonical_face()) / _hw(mean_frontal)


def fit_similarity_batched(coords, anchor_idx, Q_ref):
    N, V, D = coords.shape
    P = coords[:, anchor_idx, :]
    P_mean = P.mean(axis=1, keepdims=True); Q_mean = Q_ref.mean(axis=0, keepdims=True)
    P_c, Q_c = P - P_mean, Q_ref - Q_mean
    H = np.einsum("ki,nkj->nij", Q_c, P_c)
    U, S, Vt = np.linalg.svd(H)
    dsign = np.sign(np.linalg.det(U @ Vt))
    Dm = np.broadcast_to(np.eye(D), (N, D, D)).copy(); Dm[:, -1, -1] = dsign
    R = U @ Dm @ Vt
    var_P = (P_c ** 2).sum(axis=(1, 2))
    sign_vec = np.ones((N, D)); sign_vec[:, -1] = dsign
    s = (S * sign_vec).sum(axis=1) / np.maximum(var_P, 1e-12)
    t = Q_mean - s[:, None] * np.einsum("nij,nj->ni", R, P_mean[:, 0, :])
    return s[:, None, None] * np.einsum("nvi,nji->nvj", coords, R) + t[:, None, :]


def gpa_align(coords, anchor_idx):
    Q_ref = coords[0, anchor_idx, :].astype(np.float64).copy()
    aligned = coords
    for it in range(GPA_MAX_ITER):
        aligned = fit_similarity_batched(coords, anchor_idx, Q_ref)
        new_ref = aligned[:, anchor_idx, :].mean(axis=0)
        change = np.linalg.norm(new_ref - Q_ref) / max(np.linalg.norm(Q_ref), 1e-9)
        Q_ref = new_ref
        if it > 0 and change < GPA_TOL:
            aligned = fit_similarity_batched(coords, anchor_idx, Q_ref); break
    resid = np.linalg.norm(aligned[:, anchor_idx, :] - Q_ref[None], axis=2).max(axis=1)
    keep = resid <= np.percentile(resid, 99.0)
    return aligned, Q_ref, keep


def make_X(df, with_pose):
    au = df[AU20].to_numpy(np.float64)
    if not with_pose:
        return au
    pose = df[POSE].to_numpy(np.float64)
    inter = np.empty((len(df), len(POSE) * len(AU20)))
    k = 0
    for j in range(len(POSE)):
        for i in range(len(AU20)):
            inter[:, k] = pose[:, j] * au[:, i]; k += 1
    return np.hstack([au, pose, inter])


def linfit(X, Y, n_deploy):
    pls = PLSRegression(n_components=min(K, X.shape[1]), scale=True, max_iter=2000, tol=1e-6)
    pls.fit(X, Y)
    nf = X.shape[1]
    intercept = pls.predict(np.zeros((1, nf)))[0]
    coef_full = pls.predict(np.eye(nf)) - intercept[None, :]
    return coef_full[:n_deploy].astype(np.float32), intercept.astype(np.float32)


def remove_video_mean(aligned, vids):
    A = aligned.copy()
    for v in pd.unique(vids):
        m = vids == v
        A[m] = A[m] - A[m].mean(axis=0, keepdims=True)
    return A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/Storage/Projects/mp_blendshapes/data/pls_v24")
    ap.add_argument("--limit", type=int, default=None, help="cap rows (debug)")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    dfs, meshes = [], []
    for s in ("a", "b"):
        d = pd.read_parquet(f"{CHIPS}/v24chips_celebvhq_{s}_au_pose.parquet")
        m = np.load(f"{CHIPS}/v24chips_celebvhq_{s}_mesh.npy")          # [N,478,3] f16
        mask = (d.Yaw.abs() <= YAW_LIMIT) & (d.Pitch.abs() <= PITCH_LIMIT) \
            & d[AU20 + POSE].notna().all(axis=1)
        mask = mask.to_numpy()
        dfs.append(d[mask].reset_index(drop=True))
        meshes.append(m[mask].astype(np.float64))
        print(f"[{s}] {int(mask.sum())}/{len(d)} frames after pose-filter", flush=True)
    df = pd.concat(dfs, ignore_index=True)
    mesh = np.concatenate(meshes, axis=0)
    if args.limit:
        df = df.iloc[:args.limit].copy(); mesh = mesh[:args.limit]
    print(f"[fit] {len(df)} frames / {df.video_id.nunique()} videos", flush=True)

    aligned, ref, keep = gpa_align(mesh, MESH_ANCHORS)
    print(f"[gpa] kept {int(keep.sum())}/{len(keep)} (99th-pct anchor-residual filter)", flush=True)

    vids = df["video_id"].to_numpy()[keep]
    A = aligned[keep]; V = A.shape[1]
    # Orient to the canonical MediaPipe face (GPA seed pose is arbitrary). Procrustes
    # to canonical sets yaw/roll/pitch to the natural frontal pose.
    R = align_to_canonical(A.mean(axis=0))
    A = A @ R.T
    ref = ref @ R.T
    print(f"[orient] Procrustes-aligned aligned frame to canonical face", flush=True)
    # Aspect correction: the square chip warps the (taller) RetinaFace bbox, so
    # the model mesh is ~1.26x vertically compressed. Scale y to the canonical
    # MediaPipe face H/W so plotted faces are anatomically proportioned.
    sy = aspect_scale_y(A.mean(axis=0))
    A[:, :, 1] *= sy
    ref[:, 1] *= sy
    print(f"[aspect] scaled mesh-y by {sy:.3f} to canonical H/W", flush=True)
    mean_aligned = A.mean(axis=0)
    Yabs = np.concatenate([A[:, :, d] for d in range(3)], axis=1)
    Ydelta = np.concatenate([remove_video_mean(A, vids)[:, :, d] for d in range(3)], axis=1)
    dfk = df.iloc[np.where(keep)[0]]

    for cond in CONDITIONS:
        with_pose = cond != "nopose"
        Y = Ydelta if cond == "noid" else Yabs
        X = make_X(dfk, with_pose)
        n_deploy = (len(AU20) + len(POSE)) if with_pose else len(AU20)
        coef, intercept = linfit(X, Y, n_deploy)
        # train R2
        pred = X[:, :n_deploy] @ coef + intercept
        ss_res = ((Y - pred) ** 2).sum(); ss_tot = ((Y - Y.mean(0)) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        out = os.path.join(args.out_dir, f"mesh_{cond}.npz")
        np.savez(out, coef=coef, intercept=intercept,
                 au_columns=np.array(AU20),
                 pose_columns=(np.array(POSE) if with_pose else np.array([])),
                 mean_aligned=mean_aligned.astype(np.float32),
                 anchor_indices=np.array(MESH_ANCHORS, np.int32),
                 reference_anchors=ref.astype(np.float32),
                 ndim=np.int32(3), n_verts=np.int32(V),
                 is_delta=np.bool_(cond == "noid"), with_pose=np.bool_(with_pose))
        print(f"  wrote {out}  coef{coef.shape} n={X.shape[0]} cond={cond} trainR2={r2:.4f}", flush=True)


if __name__ == "__main__":
    main()
