"""L2CS-Net gaze estimator (MIT, Abdelrahman et al. 2022)."""
from feat.gaze_detectors.l2cs.l2cs_model import (
    L2CS,
    L2CSPipeline,
    l2cs_resnet18,
    l2cs_resnet50,
)


def load_l2cs_from_hf(
    device: str = "cpu",
    repo_id: str = "py-feat/l2cs",
    filename: str = "l2cs_gaze360_resnet50.safetensors",
    backbone: str = "resnet50",
) -> L2CSPipeline:
    """Download L2CS weights from HuggingFace and return a ready pipeline.

    Expects safetensors weights at ``<repo_id>/<filename>``. Avoids
    upstream's pickle format on the inference path; see
    ``scripts/convert_l2cs_pickle_to_safetensors.py`` for the one-time
    pickle → safetensors conversion that produces the artifact. Cached
    under py-feat's standard resource path.
    """
    from huggingface_hub import hf_hub_download
    from safetensors.torch import load_file
    from feat.utils.io import get_resource_path

    path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=get_resource_path())
    if backbone == "resnet50":
        model = l2cs_resnet50()
    elif backbone == "resnet18":
        model = l2cs_resnet18()
    else:
        raise ValueError(f"unsupported L2CS backbone: {backbone!r}")
    state = load_file(path, device="cpu")
    # Strip "module." prefix from DataParallel checkpoints if present.
    state = {k.replace("module.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return L2CSPipeline(model, device=device)


__all__ = ["L2CS", "L2CSPipeline", "l2cs_resnet18", "l2cs_resnet50", "load_l2cs_from_hf"]
