# Live benchmarks

py-feat's detector **throughput** and **accuracy**, tracked over releases.
Numbers come from `scripts/bench_detectors.py` (throughput) and
`feat.evaluation` (accuracy), are appended to the
[`py-feat/benchmarks`](https://huggingface.co/datasets/py-feat/benchmarks)
dataset by a scheduled run, and this page re-renders whenever new data lands.

```python
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
```

Live data from `py-feat/benchmarks`, updated each scheduled run.

## Throughput by detector and hardware

End-to-end frames per second for the **v1 `Detector`** (img2pose and
retinaface face models) and **v2 `Detectorv2`**, on the shared test video.
Left panel is single-frame (batch 1); right is batch 16. Bars are grouped
by hardware — fps is only comparable within the same device.

```python
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
```

<div class="marimo-book-output">
<div class="marimo-book-plotly" data-config="{}" data-figure='{"data":[{"alignmentgroup":"True","hovertemplate":"hardware=CPU\u003cbr\u003epanel=batch 1\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"CPU","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"CPU","offsetgroup":"CPU","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"AAAAAAAA+D/NzMzMzMwQQM3MzMzMzCFA"},"xaxis":"x","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"hardware=CPU\u003cbr\u003epanel=batch 16\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"CPU","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"CPU","offsetgroup":"CPU","orientation":"h","showlegend":false,"textposition":"auto","x":{"dtype":"f8","bdata":"zczMzMzM/D+amZmZmZkkQJqZmZmZmTFA"},"xaxis":"x2","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y2","type":"bar"},{"alignmentgroup":"True","hovertemplate":"hardware=NVIDIA RTX PRO 6000 Blackwell Workstation Edition\u003cbr\u003epanel=batch 1\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","offsetgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"ZmZmZmZmJ0AzMzMzMzM0QM3MzMzMzEJA"},"xaxis":"x","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"hardware=NVIDIA RTX PRO 6000 Blackwell Workstation Edition\u003cbr\u003epanel=batch 16\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","offsetgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","orientation":"h","showlegend":false,"textposition":"auto","x":{"dtype":"f8","bdata":"MzMzMzOzP0BmZmZmZuZkQAAAAAAAOHZA"},"xaxis":"x2","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y2","type":"bar"},{"alignmentgroup":"True","hovertemplate":"hardware=NVIDIA GeForce RTX 3090\u003cbr\u003epanel=batch 1\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA GeForce RTX 3090","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"NVIDIA GeForce RTX 3090","offsetgroup":"NVIDIA GeForce RTX 3090","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"mpmZmZmZKEAAAAAAAAA0QDMzMzMz80JA"},"xaxis":"x","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"hardware=NVIDIA GeForce RTX 3090\u003cbr\u003epanel=batch 16\u003cbr\u003eframes / sec (long test video)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA GeForce RTX 3090","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"NVIDIA GeForce RTX 3090","offsetgroup":"NVIDIA GeForce RTX 3090","orientation":"h","showlegend":false,"textposition":"auto","x":{"dtype":"f8","bdata":"AAAAAAAAN0BmZmZmZoZYQAAAAAAAUGlA"},"xaxis":"x2","y":["Detector · img2pose","Detector · retinaface","Detectorv2"],"yaxis":"y2","type":"bar"}],"layout":{"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,0.49],"title":{"text":"frames / sec (long test video)"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":""}},"xaxis2":{"anchor":"y2","domain":[0.51,1.0],"matches":"x","title":{"text":"frames / sec (long test video)"}},"yaxis2":{"anchor":"x2","domain":[0.0,1.0],"matches":"y","showticklabels":false},"annotations":[{"font":{},"showarrow":false,"text":"batch 1","x":0.245,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"},{"font":{},"showarrow":false,"text":"batch 16","x":0.755,"xanchor":"center","xref":"paper","y":1.0,"yanchor":"bottom","yref":"paper"}],"legend":{"title":{"text":""},"tracegroupgap":0},"title":{"text":"End-to-end detector throughput — py-feat v0.7.0"},"barmode":"group","width":900,"height":420,"dragmode":"zoom"}}'></div>
</div>

## Accuracy

*Coming in Phase 2* — per-AU F1, emotion accuracy, and valence/arousal CCC
on the held-out benchmark datasets (DISFA+, AffectNet, RAF-DB, AffWild2),
read from `accuracy.csv` in the same dataset and plotted over releases.
