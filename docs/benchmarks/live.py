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

    _REPO = "py-feat/benchmarks"

    def _load(fname):
        from huggingface_hub import hf_hub_download

        return pd.read_csv(hf_hub_download(_REPO, fname, repo_type="dataset"))

    def _placeholder():
        # Rendered until py-feat/benchmarks is populated; the real data replaces it
        # automatically once the dataset exists. Columns match the live schema.
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

    try:
        throughput = _load("throughput.csv")
    except Exception:
        throughput = pd.DataFrame()
    is_sample = throughput.empty
    if is_sample:
        throughput = _placeholder()
    return is_sample, pd, throughput


@app.cell(hide_code=True)
def _(is_sample, mo):
    mo.md(
        "> ⚠️ **Placeholder data** — the `py-feat/benchmarks` dataset isn't "
        "populated yet, so the figures below use sample numbers to show the layout."
        if is_sample
        else "Live data from `py-feat/benchmarks`, updated each scheduled run."
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ## Throughput over releases

        Frames per second per detector config, on fixed hardware. Each line is one
        **(host, GPU, config)** — fps is only comparable within the same hardware,
        so series are never mixed across machines.
        """
    )
    return


@app.cell
def _(throughput):
    import plotly.express as px

    df = throughput.copy()
    if "section_kind" in df.columns:
        df = df[df["section_kind"] == "video"]
    df["series"] = df["host"].astype(str)
    if "gpu" in df.columns:
        df["series"] = df["series"] + " · " + df["gpu"].astype(str)

    fig = px.line(
        df.sort_values("feat_version"),
        x="feat_version",
        y="fps",
        color="config",
        line_dash="series" if df["series"].nunique() > 1 else None,
        markers=True,
        labels={"feat_version": "py-feat version", "fps": "frames / sec"},
        title="Detector throughput by release",
    )
    fig.update_layout(width=760, height=460, legend_title_text="")
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
