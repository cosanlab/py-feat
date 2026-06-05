# Live benchmarks

py-feat's detector **throughput** and **accuracy**, tracked over releases.
Numbers come from `scripts/bench_detectors.py` (throughput) and
`feat.evaluation` (accuracy), are appended to the
[`py-feat/benchmarks`](https://huggingface.co/datasets/py-feat/benchmarks)
dataset by a scheduled run, and this page re-renders whenever new data lands.

```python
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
```

> ⚠️ **Placeholder data** — the `py-feat/benchmarks` dataset isn't populated yet, so the figures below use sample numbers to show the layout.

## Throughput over releases

Frames per second per detector config, on fixed hardware. Each line is one
**(host, GPU, config)** — fps is only comparable within the same hardware,
so series are never mixed across machines.

```python
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
```

<div class="marimo-book-output">
<div class="marimo-book-plotly" data-config="{}" data-figure='{"data":[{"hovertemplate":"config=Detectorv2 multitask\u003cbr\u003epy-feat version=%{x}\u003cbr\u003eframes / sec=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"Detectorv2 multitask","line":{"color":"#636efa","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines+markers","name":"Detectorv2 multitask","orientation":"v","showlegend":true,"x":["0.7.0","0.7.1","2.0.0"],"xaxis":"x","y":{"dtype":"f8","bdata":"AAAAAAAAKkDNzMzMzMwtQAAAAAAAgDNA"},"yaxis":"y","type":"scatter"},{"hovertemplate":"config=retinaface\u003cbr\u003epy-feat version=%{x}\u003cbr\u003eframes / sec=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"retinaface","line":{"color":"#EF553B","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines+markers","name":"retinaface","orientation":"v","showlegend":true,"x":["0.7.0","0.7.1","2.0.0"],"xaxis":"x","y":{"dtype":"f8","bdata":"AAAAAAAAIkCamZmZmZkkQAAAAAAAACtA"},"yaxis":"y","type":"scatter"},{"hovertemplate":"config=img2pose\u003cbr\u003epy-feat version=%{x}\u003cbr\u003eframes / sec=%{y}\u003cextra\u003e\u003c/extra\u003e","legendgroup":"img2pose","line":{"color":"#00cc96","dash":"solid"},"marker":{"symbol":"circle"},"mode":"lines+markers","name":"img2pose","orientation":"v","showlegend":true,"x":["0.7.0","0.7.1","2.0.0"],"xaxis":"x","y":{"dtype":"f8","bdata":"AAAAAAAAFEAzMzMzMzMXQAAAAAAAAB5A"},"yaxis":"y","type":"scatter"}],"layout":{"template":{"data":{"histogram2dcontour":[{"type":"histogram2dcontour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"choropleth":[{"type":"choropleth","colorbar":{"outlinewidth":0,"ticks":""}}],"histogram2d":[{"type":"histogram2d","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"heatmap":[{"type":"heatmap","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"contourcarpet":[{"type":"contourcarpet","colorbar":{"outlinewidth":0,"ticks":""}}],"contour":[{"type":"contour","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"surface":[{"type":"surface","colorbar":{"outlinewidth":0,"ticks":""},"colorscale":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]]}],"mesh3d":[{"type":"mesh3d","colorbar":{"outlinewidth":0,"ticks":""}}],"scatter":[{"fillpattern":{"fillmode":"overlay","size":10,"solidity":0.2},"type":"scatter"}],"parcoords":[{"type":"parcoords","line":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolargl":[{"type":"scatterpolargl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"bar":[{"error_x":{"color":"#2a3f5f"},"error_y":{"color":"#2a3f5f"},"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"bar"}],"scattergeo":[{"type":"scattergeo","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterpolar":[{"type":"scatterpolar","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"histogram":[{"marker":{"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"histogram"}],"scattergl":[{"type":"scattergl","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatter3d":[{"type":"scatter3d","line":{"colorbar":{"outlinewidth":0,"ticks":""}},"marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermap":[{"type":"scattermap","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattermapbox":[{"type":"scattermapbox","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scatterternary":[{"type":"scatterternary","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"scattercarpet":[{"type":"scattercarpet","marker":{"colorbar":{"outlinewidth":0,"ticks":""}}}],"carpet":[{"aaxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"baxis":{"endlinecolor":"#2a3f5f","gridcolor":"white","linecolor":"white","minorgridcolor":"white","startlinecolor":"#2a3f5f"},"type":"carpet"}],"table":[{"cells":{"fill":{"color":"#EBF0F8"},"line":{"color":"white"}},"header":{"fill":{"color":"#C8D4E3"},"line":{"color":"white"}},"type":"table"}],"barpolar":[{"marker":{"line":{"color":"#E5ECF6","width":0.5},"pattern":{"fillmode":"overlay","size":10,"solidity":0.2}},"type":"barpolar"}],"pie":[{"automargin":true,"type":"pie"}]},"layout":{"autotypenumbers":"strict","colorway":["#636efa","#EF553B","#00cc96","#ab63fa","#FFA15A","#19d3f3","#FF6692","#B6E880","#FF97FF","#FECB52"],"font":{"color":"#2a3f5f"},"hovermode":"closest","hoverlabel":{"align":"left"},"paper_bgcolor":"white","plot_bgcolor":"#E5ECF6","polar":{"bgcolor":"#E5ECF6","angularaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"radialaxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"ternary":{"bgcolor":"#E5ECF6","aaxis":{"gridcolor":"white","linecolor":"white","ticks":""},"baxis":{"gridcolor":"white","linecolor":"white","ticks":""},"caxis":{"gridcolor":"white","linecolor":"white","ticks":""}},"coloraxis":{"colorbar":{"outlinewidth":0,"ticks":""}},"colorscale":{"sequential":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"sequentialminus":[[0.0,"#0d0887"],[0.1111111111111111,"#46039f"],[0.2222222222222222,"#7201a8"],[0.3333333333333333,"#9c179e"],[0.4444444444444444,"#bd3786"],[0.5555555555555556,"#d8576b"],[0.6666666666666666,"#ed7953"],[0.7777777777777778,"#fb9f3a"],[0.8888888888888888,"#fdca26"],[1.0,"#f0f921"]],"diverging":[[0,"#8e0152"],[0.1,"#c51b7d"],[0.2,"#de77ae"],[0.3,"#f1b6da"],[0.4,"#fde0ef"],[0.5,"#f7f7f7"],[0.6,"#e6f5d0"],[0.7,"#b8e186"],[0.8,"#7fbc41"],[0.9,"#4d9221"],[1,"#276419"]]},"xaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"yaxis":{"gridcolor":"white","linecolor":"white","ticks":"","title":{"standoff":15},"zerolinecolor":"white","automargin":true,"zerolinewidth":2},"scene":{"xaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"yaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2},"zaxis":{"backgroundcolor":"#E5ECF6","gridcolor":"white","linecolor":"white","showbackground":true,"ticks":"","zerolinecolor":"white","gridwidth":2}},"shapedefaults":{"line":{"color":"#2a3f5f"}},"annotationdefaults":{"arrowcolor":"#2a3f5f","arrowhead":0,"arrowwidth":1},"geo":{"bgcolor":"white","landcolor":"#E5ECF6","subunitcolor":"white","showland":true,"showlakes":true,"lakecolor":"white"},"title":{"x":0.05},"mapbox":{"style":"light"}}},"xaxis":{"anchor":"y","domain":[0.0,1.0],"title":{"text":"py-feat version"}},"yaxis":{"anchor":"x","domain":[0.0,1.0],"title":{"text":"frames / sec"}},"legend":{"title":{"text":""},"tracegroupgap":0},"title":{"text":"Detector throughput by release"},"width":760,"height":460,"dragmode":"zoom"}}'></div>
</div>

## Accuracy

*Coming in Phase 2* — per-AU F1, emotion accuracy, and valence/arousal CCC
on the held-out benchmark datasets (DISFA+, AffectNet, RAF-DB, AffWild2),
read from `accuracy.csv` in the same dataset and plotted over releases.
