import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from detector import FaceDetector
from embedder import FaceEmbedder
from pipeline import FacePipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate face verification with FAR/FRR/EER from image pairs."
    )
    parser.add_argument("--pairs-csv", required=True, help="CSV with columns: img1,img2,label[,group]")
    parser.add_argument("--output-dir", default="face_pipeline/evaluation/out", help="Output folder")
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--decision-threshold", type=float, default=None, help="Threshold for group-wise FAR/FRR. If omitted, EER threshold is used.")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def read_pairs(csv_path):
    rows = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"img1", "img2", "label"}
        if not required.issubset(set(reader.fieldnames or [])):
            raise ValueError("pairs CSV must include columns: img1,img2,label")
        for row in reader:
            label = int(row["label"])
            if label not in (0, 1):
                raise ValueError(f"label must be 0 or 1, got: {row['label']}")
            rows.append(
                {
                    "img1": row["img1"],
                    "img2": row["img2"],
                    "label": label,
                    "group": row.get("group", "all") or "all",
                }
            )
    return rows


def compute_scores(rows, pipeline):
    results = []
    skipped = []

    for index, row in enumerate(rows):
        img1_path = Path(row["img1"])
        img2_path = Path(row["img2"])

        if not img1_path.exists() or not img2_path.exists():
            skipped.append({"index": index, "reason": "missing_file", **row})
            continue

        image1 = Image.open(img1_path).convert("RGB")
        image2 = Image.open(img2_path).convert("RGB")

        face1 = pipeline.detect_face(image1)
        face2 = pipeline.detect_face(image2)

        if face1 is None or face2 is None:
            skipped.append({"index": index, "reason": "face_not_detected", **row})
            continue

        emb1 = pipeline.get_embedding(face1)
        emb2 = pipeline.get_embedding(face2)
        similarity = pipeline.cosine_similarity(emb1, emb2)

        results.append({"similarity": similarity, **row})

    return results, skipped


def rates_at_threshold(scores, labels, threshold):
    scores = np.array(scores)
    labels = np.array(labels)

    predicted_same = scores >= threshold
    positives = labels == 1
    negatives = labels == 0

    false_accepts = np.sum(predicted_same & negatives)
    false_rejects = np.sum((~predicted_same) & positives)

    num_negatives = np.sum(negatives)
    num_positives = np.sum(positives)

    far = (false_accepts / num_negatives) if num_negatives > 0 else 0.0
    frr = (false_rejects / num_positives) if num_positives > 0 else 0.0
    return far, frr


def compute_curves(results, threshold_min, threshold_max, threshold_step):
    if not results:
        raise ValueError("No valid pairs to evaluate after filtering/skips.")

    scores = [r["similarity"] for r in results]
    labels = [r["label"] for r in results]

    thresholds = np.arange(threshold_min, threshold_max + threshold_step, threshold_step)
    fars = []
    frrs = []

    for threshold in thresholds:
        far, frr = rates_at_threshold(scores, labels, threshold)
        fars.append(far)
        frrs.append(frr)

    fars = np.array(fars)
    frrs = np.array(frrs)
    idx = int(np.argmin(np.abs(fars - frrs)))

    eer_threshold = float(thresholds[idx])
    eer = float((fars[idx] + frrs[idx]) / 2.0)

    return thresholds, fars, frrs, eer_threshold, eer


def compute_group_metrics(results, threshold):
    grouped = {}
    for row in results:
        grouped.setdefault(row["group"], []).append(row)

    metrics = []
    for group, rows in grouped.items():
        scores = [r["similarity"] for r in rows]
        labels = [r["label"] for r in rows]
        far, frr = rates_at_threshold(scores, labels, threshold)
        metrics.append(
            {
                "group": group,
                "num_pairs": len(rows),
                "far": float(far),
                "frr": float(frr),
            }
        )

    return sorted(metrics, key=lambda x: x["group"])


def save_outputs(output_dir, thresholds, fars, frrs, summary, group_metrics, skipped):
    output_dir.mkdir(parents=True, exist_ok=True)

    curve_csv = output_dir / "threshold_metrics.csv"
    with open(curve_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "far", "frr"])
        for threshold, far, frr in zip(thresholds, fars, frrs):
            writer.writerow([float(threshold), float(far), float(frr)])

    group_csv = output_dir / "group_metrics.csv"
    with open(group_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "num_pairs", "far", "frr"])
        writer.writeheader()
        writer.writerows(group_metrics)

    skipped_csv = output_dir / "skipped_pairs.csv"
    with open(skipped_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["index", "reason", "img1", "img2", "label", "group"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(skipped)

    summary_json = output_dir / "metrics_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.axvline(summary["eer_threshold"], linestyle="--", label=f"EER threshold={summary['eer_threshold']:.3f}")
    plt.title("FAR/FRR vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve.png", dpi=150)
    plt.close()


def main():
    args = parse_args()

    rows = read_pairs(args.pairs_csv)

    detector = FaceDetector(device=args.device)
    embedder = FaceEmbedder(device=args.device)
    pipeline = FacePipeline(detector, embedder)

    results, skipped = compute_scores(rows, pipeline)
    thresholds, fars, frrs, eer_threshold, eer = compute_curves(
        results,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )

    decision_threshold = args.decision_threshold if args.decision_threshold is not None else eer_threshold
    group_metrics = compute_group_metrics(results, decision_threshold)

    summary = {
        "total_pairs": len(rows),
        "evaluated_pairs": len(results),
        "skipped_pairs": len(skipped),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "decision_threshold": float(decision_threshold),
    }

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, thresholds, fars, frrs, summary, group_metrics, skipped)

    print("Evaluation complete")
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
