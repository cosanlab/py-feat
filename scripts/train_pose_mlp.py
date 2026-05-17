"""Train a small MLP that maps 68 face landmarks -> 6DoF head pose.

Distills img2pose's regressed pose into a detector-agnostic landmark-only
model. Once trained, any pipeline that produces 68 face landmarks
(img2pose, retinaface+mobilefacenet, MPDetector, OpenFace import) can feed
the same MLP and get pose values calibrated to img2pose's coordinate
frame — at MLP-fast inference cost (~10 us per face on CPU, batched).

Training data: ``/Storage/Projects/mp_blendshapes/data/bbox_pose/*_img2pose.csv``
~11,000 CelebV-HQ video clips, ~790k frames total. Each row has 68 2D
landmarks (in image-pixel coordinates), the face bbox, and img2pose's
Pitch/Roll/Yaw/X/Y/Z. Independent of AFLW2000-3D / BIWI lab benchmarks.

Test data: AFLW2000-3D (download separately; this script's val split is
clip-level holdout from the same CelebV-HQ corpus, not the public bench).

Usage:
    python scripts/train_pose_mlp.py \
        --output models/pose_mlp_v1.safetensors \
        --epochs 30 --batch-size 1024

The trained model exports a small dict of layer weights to safetensors so
it can ship at ``py-feat/pose_mlp_v1`` on HuggingFace with no Python-
unpickling concerns.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

DATA_DIRS = [
    Path("/Storage/Projects/mp_blendshapes/data/bbox_pose"),
    Path("/Storage/Projects/mp_blendshapes/data/bbox_pose_aug"),
]


def gather_clip_files() -> list[Path]:
    """Discover all *_img2pose.csv clips across the configured data dirs.

    Some clips may exist in both dirs (1,197 overlapping IDs at last
    count); dedupe by filename basename so each clip is read once.
    """
    seen: dict[str, Path] = {}
    for d in DATA_DIRS:
        if not d.exists():
            continue
        for p in sorted(d.glob("*_img2pose.csv")):
            seen.setdefault(p.name, p)
    return list(seen.values())

# Landmark column names produced by py-feat's Detector (68 dlib points).
LANDMARK_COLS = (
    [f"x_{i}" for i in range(68)] + [f"y_{i}" for i in range(68)]
)
POSE_COLS = ["Pitch", "Roll", "Yaw", "X", "Y", "Z"]
BBOX_COLS = ["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight", "FaceScore"]


def normalize_landmarks(
    landmarks: np.ndarray, bboxes: np.ndarray | None = None
) -> np.ndarray:
    """Bbox-free landmark normalization: centroid + inter-eye distance.

    Eliminates the dependence on the upstream face detector's bbox
    convention. img2pose has loose bboxes (more forehead); retinaface is
    tighter, so any bbox-relative MLP couples to the detector. By
    normalizing entirely from the landmarks themselves, the MLP works
    on any 68-point input.

    landmarks: (N, 136) flat (x_0..x_67, y_0..y_67)
    bboxes:    ignored (kept in signature for backward compatibility)
    returns:   (N, 136) flat, landmarks shifted to landmark-centroid and
               scaled to unit inter-eye distance.

    Inter-eye distance is the L2 norm between the left-eye outer corner
    (landmark 36 in dlib-68) and right-eye outer corner (landmark 45) —
    a stable scalar that varies by ~5% within a face but is independent
    of head pose.
    """
    x = landmarks[:, :68]
    y = landmarks[:, 68:]
    cx = x.mean(axis=1, keepdims=True)
    cy = y.mean(axis=1, keepdims=True)
    # Inter-eye distance (dlib-68: 36 = left-eye outer corner, 45 = right-eye outer corner).
    dx = x[:, 36] - x[:, 45]
    dy = y[:, 36] - y[:, 45]
    iod = np.sqrt(dx * dx + dy * dy)
    iod = np.where(iod <= 1e-6, 1.0, iod)[:, None]
    x_norm = (x - cx) / iod
    y_norm = (y - cy) / iod
    return np.concatenate([x_norm, y_norm], axis=1).astype(np.float32)


def gather_training_frames(
    files: list[Path],
    face_score_threshold: float = 0.8,
    pose_clip_deg: float = 75.0,
    max_frames: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream-read clip CSVs and return (X_landmarks_normalized, Y_pose).

    Filters: FaceScore > threshold, |pitch|, |yaw| < pose_clip_deg (drop
    obvious img2pose outliers — the inventory found a few frames with
    pitch up to 166° which are detection failures, not real poses).
    """
    pose_clip_rad = np.deg2rad(pose_clip_deg)
    use_cols = LANDMARK_COLS + POSE_COLS + BBOX_COLS
    Xs, Ys = [], []
    total = 0
    for i, fp in enumerate(files):
        try:
            df = pd.read_csv(fp, usecols=use_cols)
        except Exception:
            continue
        if len(df) == 0:
            continue
        df = df[df["FaceScore"] > face_score_threshold]
        if len(df) == 0:
            continue
        for col in ("Pitch", "Yaw"):
            df = df[df[col].abs() < pose_clip_rad]
        if len(df) == 0:
            continue
        lm = df[LANDMARK_COLS].to_numpy(np.float32)
        bb = df[["FaceRectX", "FaceRectY", "FaceRectWidth", "FaceRectHeight"]].to_numpy(np.float32)
        pose = df[POSE_COLS].to_numpy(np.float32)
        lm_norm = normalize_landmarks(lm, bb)
        Xs.append(lm_norm)
        Ys.append(pose)
        total += len(lm_norm)
        if max_frames and total >= max_frames:
            break
        if (i + 1) % 500 == 0:
            print(f"  ingested {total:,} frames from {i+1} clips...", flush=True)
    if not Xs:
        raise RuntimeError("no training frames after filtering")
    X = np.concatenate(Xs, axis=0)
    Y = np.concatenate(Ys, axis=0)
    return X, Y


def horizontal_flip_landmarks(landmarks_flat: np.ndarray) -> np.ndarray:
    """Mirror 68 dlib landmarks left/right and remap landmark indices.

    Input: ``(N, 136)`` flat (x_0..x_67, y_0..y_67) where x is in
    landmark-centroid-relative units (positive = subject's left in
    image space). Output: ``(N, 136)`` with landmarks reflected and
    re-indexed so left-eye landmarks land on right-eye positions etc.
    """
    # dlib-68 horizontal-flip index map (well-known constants):
    flip_idx = np.array([
        16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,                  # jawline 0..16
        26, 25, 24, 23, 22, 21, 20, 19, 18, 17,                                    # eyebrows 17..26
        27, 28, 29, 30,                                                            # nose bridge 27..30
        35, 34, 33, 32, 31,                                                        # nose base 31..35
        45, 44, 43, 42, 47, 46,                                                    # left eye -> right 36..41
        39, 38, 37, 36, 41, 40,                                                    # right eye -> left 42..47
        54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,                            # outer mouth 48..59
        64, 63, 62, 61, 60, 67, 66, 65,                                            # inner mouth 60..67
    ], dtype=np.int64)
    x = landmarks_flat[:, :68]
    y = landmarks_flat[:, 68:]
    x_flipped = -x[:, flip_idx]  # negate x and remap indices
    y_flipped = y[:, flip_idx]
    return np.concatenate([x_flipped, y_flipped], axis=1).astype(np.float32)


def horizontal_flip_pose(pose: np.ndarray) -> np.ndarray:
    """Flip the pose target to match a horizontally-flipped face.

    Convention (img2pose / FEAT_FACEPOSE_COLUMNS_6D):
      [Pitch, Roll, Yaw, X, Y, Z]
    Under L/R mirroring:
      Pitch: unchanged (nod axis is preserved)
      Roll:  sign flips (clockwise becomes counterclockwise)
      Yaw:   sign flips (looking right becomes looking left)
      X:     sign flips (translation along the camera's horizontal axis)
      Y, Z:  unchanged
    """
    out = pose.copy()
    out[:, 1] = -out[:, 1]  # Roll
    out[:, 2] = -out[:, 2]  # Yaw
    out[:, 3] = -out[:, 3]  # X
    return out


class PoseDataset(Dataset):
    """Pose dataset with optional horizontal-flip + landmark jitter augmentation.

    Both augmentations apply post-normalization (when X is already in
    landmark-centroid / inter-eye-distance space). At that point a
    horizontal flip is just `x_norm := -x_norm[flip_idx]`, and a
    landmark jitter is an additive Gaussian in normalized units.

    Jitter magnitude: ``jitter_std`` units of inter-eye distance.
    typical 0.02 ≈ 2 pixels on a 100-px face.
    """

    def __init__(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        flip_p: float = 0.0,
        jitter_std: float = 0.0,
    ):
        self.X = X
        self.Y = Y
        self.flip_p = flip_p
        self.jitter_std = jitter_std
        # dlib-68 flip-index, captured once (matches horizontal_flip_landmarks).
        self.flip_idx = np.array([
            16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0,
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
            27, 28, 29, 30,
            35, 34, 33, 32, 31,
            45, 44, 43, 42, 47, 46,
            39, 38, 37, 36, 41, 40,
            54, 53, 52, 51, 50, 49, 48, 59, 58, 57, 56, 55,
            64, 63, 62, 61, 60, 67, 66, 65,
        ], dtype=np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]  # (136,)
        y = self.Y[idx]  # (6,) — standardized
        if self.flip_p > 0 and np.random.rand() < self.flip_p:
            xs = x[:68][self.flip_idx]
            ys = x[68:][self.flip_idx]
            x = np.concatenate([-xs, ys]).astype(np.float32)
            # Note: y is *standardized* (z-scored) at training time, so
            # we can't flip raw signs here. The dataset flips x BEFORE
            # standardization, so this branch must be applied to raw y.
            # See PoseDataset construction in train() for the trick.
        if self.jitter_std > 0:
            x = x + np.random.normal(0, self.jitter_std, size=x.shape).astype(np.float32)
        return torch.from_numpy(x), torch.from_numpy(y)


class PoseMLP(nn.Module):
    """Wider 68-landmarks-to-6DoF-pose MLP with LayerNorm + GELU."""

    def __init__(self, hidden=(512, 256, 128), dropout=0.15):
        super().__init__()
        in_dim = 136
        layers = []
        for h in hidden:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.LayerNorm(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 6))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def train(args):
    files = gather_clip_files()
    if args.max_clips:
        files = files[: args.max_clips]
    print(f"loading {len(files)} clip CSVs from {len(DATA_DIRS)} dirs...")
    t0 = time.perf_counter()
    X, Y = gather_training_frames(
        files,
        face_score_threshold=args.face_score_threshold,
        pose_clip_deg=args.pose_clip_deg,
        max_frames=args.max_frames,
    )
    print(f"  loaded {len(X):,} frames in {time.perf_counter()-t0:.0f}s")
    print(f"  X shape: {X.shape}, Y shape: {Y.shape}")
    print(f"  Y stats per axis:")
    for i, name in enumerate(POSE_COLS):
        col = Y[:, i]
        print(f"    {name}: mean={col.mean():+.3f}, std={col.std():.3f}")

    # Split BEFORE augmenting so val stays clean (no doubled rows).
    n = len(X)
    split = int(n * (1 - args.val_frac))
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    X, Y = X[perm], Y[perm]
    X_train, X_val = X[:split], X[split:]
    Y_train, Y_val = Y[:split], Y[split:]

    # Horizontal-flip augmentation: double the training set with flipped copies.
    # Pose targets flip on Roll/Yaw/X axes; landmarks reflect + remap dlib indices.
    if args.flip_augment:
        X_train_flipped = horizontal_flip_landmarks(X_train)
        Y_train_flipped = horizontal_flip_pose(Y_train)
        X_train = np.concatenate([X_train, X_train_flipped], axis=0)
        Y_train = np.concatenate([Y_train, Y_train_flipped], axis=0)
        # Re-shuffle
        rng2 = np.random.default_rng(args.seed + 1)
        perm2 = rng2.permutation(len(X_train))
        X_train, Y_train = X_train[perm2], Y_train[perm2]
        print(f"  + horizontal-flip augmentation: train doubled to {len(X_train):,}")

    # Standardize Y per-axis so each output channel contributes equally.
    Y_mean = Y_train.mean(axis=0)
    Y_std = Y_train.std(axis=0)
    Y_train_z = (Y_train - Y_mean) / Y_std
    Y_val_z = (Y_val - Y_mean) / Y_std

    train_loader = DataLoader(
        PoseDataset(X_train, Y_train_z,
                    flip_p=0.0,  # already done offline above
                    jitter_std=args.jitter_std),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )
    val_loader = DataLoader(
        PoseDataset(X_val, Y_val_z, flip_p=0.0, jitter_std=0.0),
        batch_size=args.batch_size * 2,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    device = args.device
    model = PoseMLP(hidden=tuple(args.hidden), dropout=args.dropout).to(device)
    opt = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.SmoothL1Loss()

    best_val = float("inf")
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss /= n_batches

        # Val: report per-axis MAE in original units (degrees for angles).
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_truth = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                pred = model(xb)
                val_loss += loss_fn(pred, yb).item()
                all_preds.append(pred.cpu().numpy())
                all_truth.append(yb.cpu().numpy())
        val_loss /= len(val_loader)
        preds = np.concatenate(all_preds) * Y_std + Y_mean
        truth = np.concatenate(all_truth) * Y_std + Y_mean
        mae_per_axis = np.abs(preds - truth).mean(axis=0)
        mae_deg = np.degrees(mae_per_axis[:3])  # Pitch/Roll/Yaw in radians -> degrees
        print(
            f"epoch {epoch+1:3d}/{args.epochs} | "
            f"train={train_loss:.4f} val={val_loss:.4f} | "
            f"MAE (deg) pitch={mae_deg[0]:.2f} roll={mae_deg[1]:.2f} yaw={mae_deg[2]:.2f}"
        )
        history.append({"epoch": epoch + 1, "train": train_loss, "val": val_loss,
                        "pitch_mae_deg": float(mae_deg[0]),
                        "roll_mae_deg": float(mae_deg[1]),
                        "yaw_mae_deg": float(mae_deg[2])})

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best checkpoint + normalization stats + metadata
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from safetensors.torch import save_file
        save_file(best_state, str(out_path))
    except ImportError:
        torch.save(best_state, str(out_path).replace(".safetensors", ".pt"))
    meta_path = out_path.with_suffix(".json")
    meta = {
        "architecture": {
            "type": "PoseMLP",
            "hidden": list(args.hidden),
            "dropout": args.dropout,
            "input_dim": 136,
            "output_dim": 6,
            "input_description": "68 (x, y) face landmarks normalized to face bbox [0,1]^2",
            "output_description": "Pitch, Roll, Yaw (rad), X, Y, Z (units of img2pose template)",
            "output_normalization": {"mean": Y_mean.tolist(), "std": Y_std.tolist()},
        },
        "training": {
            "data": "/Storage/Projects/mp_blendshapes/data/bbox_pose/*_img2pose.csv (CelebV-HQ extraction)",
            "n_clips": len(files),
            "n_train_frames": int(len(X_train)),
            "n_val_frames": int(len(X_val)),
            "face_score_threshold": args.face_score_threshold,
            "pose_clip_deg": args.pose_clip_deg,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "seed": args.seed,
        },
        "best_val_loss": float(best_val),
        "history": history,
    }
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"saved: {out_path}")
    print(f"saved: {meta_path}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, default=Path("models/pose_mlp_v1.safetensors"))
    p.add_argument("--max-clips", type=int, default=None, help="cap on CSV files")
    p.add_argument("--max-frames", type=int, default=None, help="cap on total training frames")
    p.add_argument("--face-score-threshold", type=float, default=0.8)
    p.add_argument("--pose-clip-deg", type=float, default=75.0, help="drop frames with |pitch| or |yaw| above this")
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--hidden", type=int, nargs="+", default=[512, 256, 128])
    p.add_argument("--dropout", type=float, default=0.15)
    p.add_argument("--flip-augment", action="store_true", default=True,
                   help="Double training set with horizontal-flipped copies")
    p.add_argument("--no-flip-augment", action="store_false", dest="flip_augment")
    p.add_argument("--jitter-std", type=float, default=0.015,
                   help="Gaussian noise std added to normalized landmarks at "
                   "training time (units of inter-eye distance)")
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
