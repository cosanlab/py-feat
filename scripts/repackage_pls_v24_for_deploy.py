"""Repackage the v2.4 standard AU→mesh PLS into py-feat's deploy schema (v4).

Source: scripts/fit_pls_v24_mesh.py 'standard' fit (frontalized + aspect-corrected,
AU+pose -> absolute aligned coords). 20-AU Detectorv2-v2.4 space (== AU_LANDMARK_MAP
['Feat']). is_delta=False, so coef/intercept already give absolute coords — ship
unchanged (no neutral fold). Output keys match what feat.plotting.PLSAUMeshModel /
_load_pls_au_to_mesh_v2_from_hub read: coef (23,1434), intercept (1434,),
au_columns (20), pose_columns (3), mean_aligned_mesh (478,3).

Writes to feat/resources/ (local-load fallback before HF upload) and a /tmp copy
the user can `huggingface-cli upload py-feat/au_to_mesh au_to_mesh_pls_v4.npz`.

  uv run python scripts/repackage_pls_v24_for_deploy.py
"""
import os
import numpy as np

SRC = "/Storage/Projects/mp_blendshapes/data/pls_v24/mesh_standard.npz"
RESOURCES = "/home/ljchang/Github/py-feat/feat/resources/au_to_mesh_pls_v4.npz"
TMP = "/tmp/pls_v24_deploy/au_to_mesh_pls_v4.npz"
FEAT20 = ["AU01", "AU02", "AU04", "AU05", "AU06", "AU07", "AU09", "AU10", "AU11",
          "AU12", "AU14", "AU15", "AU17", "AU20", "AU23", "AU24", "AU25", "AU26",
          "AU28", "AU43"]


def main():
    w = np.load(SRC, allow_pickle=True)
    assert not bool(w["is_delta"]), "standard fit must be absolute (is_delta=False)"
    au_columns = [str(s) for s in w["au_columns"]]
    assert au_columns == FEAT20, f"au_columns drift: {au_columns}"
    coef = w["coef"].astype(np.float32)              # (23, 1434) = [20 AU | 3 pose]
    intercept = w["intercept"].astype(np.float32)    # (1434,)
    mean_mesh = w["mean_aligned"].astype(np.float32) # (478, 3)
    pose_columns = [str(s) for s in w["pose_columns"]]
    assert coef.shape[0] == len(au_columns) + len(pose_columns), coef.shape

    payload = dict(
        coef=coef, intercept=intercept,
        au_columns=np.array(au_columns), pose_columns=np.array(pose_columns),
        input_columns=np.array(au_columns + pose_columns),
        mean_aligned_mesh=mean_mesh,
        n_components=np.int32(coef.shape[0]),
        model_card=np.str_("au_to_mesh_pls_v4 (Detectorv2 v2.4, 20-AU Feat space; "
                           "standard AU+pose->absolute; frontalized + aspect-corrected)"),
    )
    for path in (RESOURCES, TMP):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, **payload)

    # verify: load back exactly as the deployed loader does (allow_pickle=False)
    z = np.load(RESOURCES, allow_pickle=False)
    assert [str(s) for s in z["au_columns"]] == FEAT20
    # deploy predict (pose zero-padded) == fit predict at pose=0, for random AU
    rng = np.random.default_rng(0)
    for t in range(5):
        au = rng.random(20).astype(np.float32) if t else np.zeros(20, np.float32)
        x = np.zeros(coef.shape[0], np.float32); x[:20] = au
        deploy = x @ z["coef"] + z["intercept"]
        fit = x @ coef + intercept
        assert float(np.abs(deploy - fit).max()) < 1e-5
    # sanity: H/W proportions of the neutral mesh
    M = mean_mesh
    hw = abs(M[10, 1] - M[152, 1]) / abs(M[234, 0] - M[454, 0])
    print(f"wrote {RESOURCES}")
    print(f"wrote {TMP}  (upload: huggingface-cli upload py-feat/au_to_mesh {os.path.basename(TMP)})")
    print(f"coef{coef.shape} au={len(au_columns)} pose={len(pose_columns)} mean_mesh{mean_mesh.shape} "
          f"neutral H/W={hw:.3f}  deploy==fit OK")


if __name__ == "__main__":
    main()
