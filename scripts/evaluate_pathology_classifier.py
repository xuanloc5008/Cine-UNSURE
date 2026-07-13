#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = [json.loads(Path(path).read_text(encoding="utf-8")) for path in sorted(glob.glob(args.predictions))]
    rows = [row for row in rows if row.get("target_pathology")]
    if not rows:
        raise ValueError("no pathology predictions with targets found")
    classes = list(rows[0]["pathology_probabilities"])
    class_to_index = {name: index for index, name in enumerate(classes)}
    target = np.asarray([class_to_index[row["target_pathology"]] for row in rows])
    predicted = np.asarray([class_to_index[row["predicted_pathology"]] for row in rows])
    probabilities = np.asarray(
        [[row["pathology_probabilities"][name]["mean"] for name in classes] for row in rows], dtype=float
    )
    one_hot = np.eye(len(classes))[target]
    per_class = {}
    f1_values = []
    recalls = []
    for index, name in enumerate(classes):
        tp = np.sum((target == index) & (predicted == index))
        fp = np.sum((target != index) & (predicted == index))
        fn = np.sum((target == index) & (predicted != index))
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": int(np.sum(target == index))}
        f1_values.append(f1)
        recalls.append(recall)

    result = {
        "num_sequences": len(rows),
        "classes": classes,
        "accuracy": float(np.mean(target == predicted)),
        "balanced_accuracy": float(np.mean(recalls)),
        "macro_f1": float(np.mean(f1_values)),
        "negative_log_likelihood": float(-np.mean(np.log(np.clip(probabilities[np.arange(len(rows)), target], 1e-8, 1.0)))),
        "multiclass_brier": float(np.mean(np.sum(np.square(probabilities - one_hot), axis=1))),
        "mean_predictive_entropy": float(np.mean(-np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1))),
        "mean_probability_band_width": float(
            np.mean(
                [
                    row["pathology_probabilities"][name]["upper"]
                    - row["pathology_probabilities"][name]["lower"]
                    for row in rows
                    for name in classes
                ]
            )
        ),
        "per_class": per_class,
        "rows": [
            {
                "source_path": row["source_path"],
                "target": row["target_pathology"],
                "prediction": row["predicted_pathology"],
            }
            for row in rows
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
