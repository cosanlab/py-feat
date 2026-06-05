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

        Frames per second per detector config (video, CUDA, largest batch). Bars are
        grouped by GPU — fps is only comparable within the same hardware. As more
        releases land this becomes a trend; for now it's the current v0.7.0 snapshot.
        """
    )
    return


@app.cell
def _(throughput):
    import plotly.express as px

    df = throughput.copy()
    for col, val in [("section_kind", "video"), ("device", "cuda")]:
        if col in df.columns:
            df = df[df[col] == val]
    if "batch" in df.columns and len(df):
        df = df[df["batch"] == df["batch"].max()]
    # One bar per (config, GPU): keep the most recent run for each.
    if {"date", "config", "gpu"}.issubset(df.columns):
        df = df.sort_values("date").drop_duplicates(["config", "gpu"], keep="last")
    if "gpu" in df.columns:
        df["GPU"] = df["gpu"].astype(str).str.replace(r"CUDA[^,]*,\s*", "", regex=True)

    fig = px.bar(
        df.sort_values("fps"),
        x="fps", y="config", color="GPU",
        orientation="h", barmode="group",
        labels={"fps": "frames / sec (video · CUDA · largest batch)", "config": ""},
        title="Detector throughput by config and GPU",
    )
    fig.update_layout(width=820, height=480, legend_title_text="")
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
