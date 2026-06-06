"""Parse the per-run throughput dumps in docs/benchmarks/*.md into a single
throughput.csv in the live-dashboard schema (docs/benchmarks/live.py).

The dumps are produced by scripts/bench_detectors.py --markdown. This collapses
them to one tidy table so the dashboard can show real numbers from a committed
CSV (and the same CSV can be pushed to the py-feat/benchmarks HF dataset).

Schema (one row per timed (config, device, batch, section) cell):
    feat_version, date, host, gpu, config, device, batch, section_kind, fps

Usage:
    python scripts/benchmarks_md_to_csv.py            # -> docs/benchmarks/throughput.csv
"""

import csv
import glob
import os
import re

BENCH_DIR = os.path.join(os.path.dirname(__file__), "..", "docs", "benchmarks")
OUT = os.path.join(BENCH_DIR, "throughput.csv")
# Only the clean, single-machine regen runs feed the dashboard (one commit,
# consistent configs). The older heterogeneous run dumps stay as history.
GLOB = "speed-clean-*.md"

# Only these ### configs carry a device|batch|fps table we want.
KNOWN_CONFIGS = {
    "img2pose": "img2pose",
    "retinaface": "retinaface",
    "MPDetector retinaface": "MPDetector retinaface",
    "Detectorv2 multitask": "Detectorv2 multitask",
}
DEVICES = {"cpu", "cuda", "mps"}


def _meta(text, label):
    m = re.search(rf"^- \*\*{re.escape(label)}:\*\*\s*(.+)$", text, re.MULTILINE)
    return m.group(1).strip() if m else ""


def section_kind(header):
    h = header.lower()
    if h.startswith("video"):
        return "video"
    if h.startswith("images"):
        return "images"
    return None  # hand-curated / non-timed section -> skip


def parse_file(path):
    text = open(path).read()
    if "py-feat version" not in text:
        return []
    version = _meta(text, "py-feat version") or "unknown"
    date = (_meta(text, "Date") or "").split(" ")[0]
    host = (_meta(text, "Host") or "").split(" ")[0]
    gpu = _meta(text, "GPU") or "cpu"

    rows, kind, label, config = [], None, "", None
    for line in text.splitlines():
        if line.startswith("## "):
            hdr = line[3:].strip()
            kind = section_kind(hdr)
            h = hdr.lower()
            label = "long" if "long" in h else "short" if "short" in h else hdr
            config = None
        elif line.startswith("### "):
            config = KNOWN_CONFIGS.get(line[4:].strip())
        elif line.startswith("|") and kind and config:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            # expect: device | batch | sec | ms/frame | fps
            if len(cells) < 5 or cells[0] not in DEVICES:
                continue
            try:
                batch = int(cells[1])
                fps = float(cells[-1])
            except ValueError:
                continue
            device = cells[0]
            # The GPU metadata names the *visible* device; CPU rows aren't on it.
            row_gpu = "CPU" if device == "cpu" else gpu
            rows.append(dict(
                feat_version=version, date=date, host=host, gpu=row_gpu,
                config=config, device=device, batch=batch,
                section_kind=kind, section_label=label, fps=fps,
            ))
    return rows


def main():
    all_rows = []
    for path in sorted(glob.glob(os.path.join(BENCH_DIR, GLOB))):
        all_rows.extend(parse_file(path))
    cols = ["feat_version", "date", "host", "gpu", "config",
            "device", "batch", "section_kind", "section_label", "fps"]
    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(all_rows)
    print(f"Wrote {len(all_rows)} rows -> {os.path.relpath(OUT)}")
    # quick summary
    by = {}
    for r in all_rows:
        by.setdefault((r["config"], r["section_kind"]), 0)
        by[(r["config"], r["section_kind"])] += 1
    for k, n in sorted(by.items()):
        print(f"  {k[0]:<24} {k[1]:<7} {n} rows")


if __name__ == "__main__":
    main()
