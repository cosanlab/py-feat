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

Showing committed benchmark data (`docs/benchmarks/throughput.csv`); the scheduled `py-feat/benchmarks` feed supersedes it once populated.

## Throughput by detector and hardware

Frames per second per detector config (video, CUDA, largest batch). Bars are
grouped by GPU — fps is only comparable within the same hardware. As more
releases land this becomes a trend; for now it's the current v0.7.0 snapshot.

```python
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
```

<div class="marimo-book-output">
<div class="marimo-book-plotly" data-config="{}" data-figure='{"data":[{"alignmentgroup":"True","hovertemplate":"GPU=NVIDIA GB10\u003cbr\u003eframes / sec (video · CUDA · largest batch)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA GB10","marker":{"color":"#636efa","pattern":{"shape":""}},"name":"NVIDIA GB10","offsetgroup":"NVIDIA GB10","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"zczMzMzMJ0AzMzMzM7M+QJqZmZmZWUFA"},"xaxis":"x","y":["img2pose","retinaface","MPDetector retinaface"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"GPU=NVIDIA GeForce RTX 3090\u003cbr\u003eframes / sec (video · CUDA · largest batch)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA GeForce RTX 3090","marker":{"color":"#EF553B","pattern":{"shape":""}},"name":"NVIDIA GeForce RTX 3090","offsetgroup":"NVIDIA GeForce RTX 3090","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"ZmZmZmbmNEDNzMzMzKxVQDMzMzMzs1VAAAAAAACgV0A="},"xaxis":"x","y":["img2pose","MPDetector retinaface","Detectorv2 multitask","retinaface"],"yaxis":"y","type":"bar"},{"alignmentgroup":"True","hovertemplate":"GPU=NVIDIA RTX PRO 6000 Blackwell Workstation Edition\u003cbr\u003eframes / sec (video · CUDA · largest batch)=%{x}\u003cbr\u003e=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","marker":{"color":"#00cc96","pattern":{"shape":""}},"name":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","offsetgroup":"NVIDIA RTX PRO 6000 Blackwell Workstation Edition","orientation":"h","showlegend":true,"textposition":"auto","x":{"dtype":"f8","bdata":"ZmZmZmZmPUCamZmZmVljQM3MzMzMXGhAmpmZmZnZcUA="},"xaxis":"x","y":["img2pose","retinaface","MPDetector retinaface","Detectorv2 multitask"],"yaxis":"y","type":"bar"}],"layout":{"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"frames / sec (video · CUDA · largest batch)"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":""}},"legend":{"title":{"text":""},"tracegroupgap":0},"title":{"text":"Detector throughput by config and GPU"},"barmode":"group","width":820,"height":480,"dragmode":"zoom"}}'></div>
</div>

## Accuracy

*Coming in Phase 2* — per-AU F1, emotion accuracy, and valence/arousal CCC
on the held-out benchmark datasets (DISFA+, AffectNet, RAF-DB, AffWild2),
read from `accuracy.csv` in the same dataset and plotted over releases.
