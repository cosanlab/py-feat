"""HuggingFace download dashboard for the py-feat org.

Lists every model under https://huggingface.co/py-feat with its monthly
and lifetime download counts, last-updated date, and license. Intended as
a maintainer tool, not part of the runtime package - run from the repo
root with `python scripts/hf_download_dashboard.py`.

Usage:
    python scripts/hf_download_dashboard.py            # default org=py-feat
    python scripts/hf_download_dashboard.py --org foo
    python scripts/hf_download_dashboard.py --json     # machine-readable output
    python scripts/hf_download_dashboard.py --csv path # write a CSV

Reads from the public HuggingFace API; no auth needed for public repos.
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass

from huggingface_hub import HfApi


@dataclass
class ModelStats:
    repo: str
    downloads_last_month: int
    downloads_all_time: int
    last_modified: str
    license: str
    tags: str


def collect(org: str) -> list[ModelStats]:
    api = HfApi()
    out: list[ModelStats] = []
    for m in api.list_models(author=org, full=True):
        info = api.model_info(m.modelId, files_metadata=False)
        license_str = (info.card_data or {}).get("license", "") if info.card_data else ""
        out.append(
            ModelStats(
                repo=info.modelId,
                downloads_last_month=getattr(info, "downloads", 0) or 0,
                downloads_all_time=getattr(info, "downloads_all_time", 0) or 0,
                last_modified=str(getattr(info, "last_modified", "") or ""),
                license=license_str or "",
                tags=", ".join(getattr(info, "tags", []) or []),
            )
        )
    out.sort(key=lambda s: s.downloads_last_month, reverse=True)
    return out


def render_table(stats: list[ModelStats]) -> str:
    rows = [
        ("Repo", "DL/mo", "DL all-time", "License", "Last modified"),
        *[
            (
                s.repo,
                f"{s.downloads_last_month:,}",
                f"{s.downloads_all_time:,}",
                s.license or "-",
                s.last_modified[:10] if s.last_modified else "-",
            )
            for s in stats
        ],
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [fmt.format(*rows[0]), fmt.format(*("-" * w for w in widths))]
    for row in rows[1:]:
        lines.append(fmt.format(*row))
    total_mo = sum(s.downloads_last_month for s in stats)
    total_all = sum(s.downloads_all_time for s in stats)
    lines.append("")
    lines.append(f"Total {len(stats)} models. Last month: {total_mo:,}. All-time: {total_all:,}.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--org", default="py-feat")
    p.add_argument("--json", action="store_true", help="emit JSON to stdout")
    p.add_argument("--csv", metavar="PATH", help="write a CSV file at PATH")
    args = p.parse_args()

    stats = collect(args.org)

    if args.csv:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(asdict(stats[0]).keys()) if stats else [])
            w.writeheader()
            for s in stats:
                w.writerow(asdict(s))
        print(f"Wrote {len(stats)} rows to {args.csv}", file=sys.stderr)

    if args.json:
        print(json.dumps([asdict(s) for s in stats], indent=2))
    else:
        print(render_table(stats))


if __name__ == "__main__":
    main()
