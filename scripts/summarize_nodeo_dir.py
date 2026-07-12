#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows_by_id: dict[str, dict] = {}
    with Path(args.summary).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows_by_id[str(row["sequence_id"])] = row
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows_by_id.values():
        grouped[str(row["dataset"])].append(row)

    report: dict[str, object] = {"summary": args.summary, "num_sequences": len(rows_by_id), "by_dataset": {}}
    metric_names = ("loss", "image", "jdet", "mag", "smooth", "fold_fraction", "abs_jdet_minus_one")
    for dataset, rows in sorted(grouped.items()):
        metrics: dict[str, dict[str, float]] = {}
        for name in metric_names:
            values = np.asarray([float(row["metrics"][name]) for row in rows], dtype=np.float64)
            metrics[name] = {
                "mean": float(values.mean()),
                "std": float(values.std()),
                "median": float(np.median(values)),
            }
        report["by_dataset"][dataset] = {"num_sequences": len(rows), "metrics": metrics}  # type: ignore[index]

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
