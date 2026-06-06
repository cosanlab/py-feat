import marimo

__generated_with = "0.23.8"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # Live benchmarks

        py-feat's detector **throughput** and **accuracy**, tracked over releases.
        Numbers come from `scripts/bench_detectors.py` (throughput) and
        `feat.evaluation` (accuracy), are appended to the
        [`py-feat/benchmarks`](https://huggingface.co/datasets/py-feat/benchmarks)
        dataset by a scheduled run, and this page re-renders whenever new data lands.
        """
    )
    return


@app.cell
def _():
    import pandas as pd
    from pathlib import Path

    _REPO = "py-feat/benchmarks"

    def _from_hub(fname):
        from huggingface_hub import hf_hub_download

        return pd.read_csv(hf_hub_download(_REPO, fname, repo_type="dataset"))

    def _from_local(fname):
        # Committed seed data next to this notebook; also the file pushed to HF.
        try:
            _base = Path(__file__).parent
        except NameError:
            _base = Path.cwd()
        p = _base / fname
        return pd.read_csv(p) if p.exists() else pd.DataFrame()

    def _placeholder():
        recs = []
        for ver, base in [("0.7.0", 1.0), ("0.7.1", 1.15), ("2.0.0", 1.5)]:
            for cfg, fps in [
                ("Detectorv2 multitask", 13.0),
                ("retinaface", 9.0),
                ("img2pose", 5.0),
            ]:
                recs.append(
                    dict(
                        feat_version=ver, date=ver, host="liquidswords2",
                        gpu="CUDA, NVIDIA (sample)", config=cfg, device="cuda",
                        batch=16, section_kind="video", fps=round(base * fps, 1),
                    )
                )
        return pd.DataFrame(recs)

    # Prefer the live HF dataset; fall back to the committed CSV; then placeholder.
    try:
        throughput = _from_hub("throughput.csv")
        source = "hf"
    except Exception:
        throughput = _from_local("throughput.csv")
        source = "committed" if not throughput.empty else "placeholder"
        if throughput.empty:
            throughput = _placeholder()
    is_sample = source == "placeholder"
    return is_sample, pd, source, throughput


@app.cell(hide_code=True)
def _(mo, source):
    mo.md(
        {
            "hf": "Live data from `py-feat/benchmarks`, updated each scheduled run.",
            "committed": "Showing committed benchmark data "
            "(`docs/benchmarks/throughput.csv`); the scheduled `py-feat/benchmarks` "
            "feed supersedes it once populated.",
            "placeholder": "> ⚠️ **Placeholder data** — the `py-feat/benchmarks` "
            "dataset isn't populated yet, so the figures below use sample numbers "
            "to show the layout.",
        }[source]
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Throughput by detector and hardware

        End-to-end frames per second for the **v1 `Detector`** (img2pose and
        retinaface face models) and **v2 `Detectorv2`**, on the shared test video.
        Left panel is single-frame (batch 1); right is batch 16. Bars are grouped
        by hardware — fps is only comparable within the same device.
        """
    )
    return


@app.cell
def _(throughput):
    import plotly.express as px

    # v1 Detector (two face models) + v2 Detectorv2; MPDetector excluded.
    _RELABEL = {
        "img2pose": "Detector · img2pose",
        "retinaface": "Detector · retinaface",
        "Detectorv2 multitask": "Detectorv2",
    }
    df = throughput.copy()
    if "section_kind" in df.columns:
        df = df[df["section_kind"] == "video"]
    if "section_label" in df.columns and (df["section_label"] == "long").any():
        df = df[df["section_label"] == "long"]
    if "batch" in df.columns:
        df = df[df["batch"].isin([1, 16])]
    df = df[df["config"].isin(_RELABEL)].copy()
    df["detector"] = df["config"].map(_RELABEL)
    df["hardware"] = [
        "CPU" if dev == "cpu" else str(g).split(",")[-1].strip()
        for dev, g in zip(df.get("device", ""), df["gpu"])
    ]
    df["panel"] = "batch " + df["batch"].astype(str)

    fig = px.bar(
        df.sort_values(["batch", "fps"]),
        x="fps", y="detector", color="hardware",
        orientation="h", barmode="group", facet_col="panel",
        labels={"fps": "frames / sec (long test video)", "detector": ""},
        title="End-to-end detector throughput — py-feat v0.7.0",
    )
    fig.update_layout(width=900, height=420, legend_title_text="")
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Accuracy

        *Coming in Phase 2* — per-AU F1, emotion accuracy, and valence/arousal CCC
        on the held-out benchmark datasets (DISFA+, AffectNet, RAF-DB, AffWild2),
        read from `accuracy.csv` in the same dataset and plotted over releases.
        """
    )
    return


if __name__ == "__main__":
    app.run()
