import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from sklearn.datasets import fetch_lfw_pairs

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from embedder import FaceEmbedder
from pipeline import FacePipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate on LFW pairs dataset")
    parser.add_argument("--subset", default="10_folds", choices=["train", "test", "10_folds"])
    parser.add_argument("--max-pairs", type=int, default=1000)
    parser.add_argument("--threshold-min", type=float, default=-1.0)
    parser.add_argument("--threshold-max", type=float, default=1.0)
    parser.add_argument("--threshold-step", type=float, default=0.01)
    parser.add_argument("--decision-threshold", type=float, default=None)
    parser.add_argument("--output-dir", default="face_pipeline/evaluation/lfw_out")
    parser.add_argument("--device", default="cpu", help="cpu or cuda")
    return parser.parse_args()


def to_pil(np_image):
    arr = np_image
    if arr.dtype.kind == "f" and np.max(arr) <= 1.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


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
    return float(far), float(frr)


def compute_curves(scores, labels, threshold_min, threshold_max, threshold_step):
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


def build_proxy_groups(num_items):
    # LFW pairs loader does not provide demographic metadata directly.
    # For now, create reproducible proxy groups by index bucket to keep
    # pipeline structure for group-wise FAR/FRR reporting.
    groups = []
    for index in range(num_items):
        bucket = index % 4
        groups.append(f"proxy_bucket_{bucket}")
    return groups


def compute_group_metrics(scores, labels, groups, threshold):
    grouped = {}
    for score, label, group in zip(scores, labels, groups):
        grouped.setdefault(group, {"scores": [], "labels": []})
        grouped[group]["scores"].append(score)
        grouped[group]["labels"].append(label)

    metrics = []
    for group, values in grouped.items():
        far, frr = rates_at_threshold(values["scores"], values["labels"], threshold)
        metrics.append(
            {
                "group": group,
                "num_pairs": len(values["scores"]),
                "far": far,
                "frr": frr,
            }
        )
    return sorted(metrics, key=lambda x: x["group"])


def save_outputs(output_dir, thresholds, fars, frrs, summary, group_metrics):
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(output_dir / "threshold_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["threshold", "far", "frr"])
        for threshold, far, frr in zip(thresholds, fars, frrs):
            writer.writerow([float(threshold), float(far), float(frr)])

    with open(output_dir / "group_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "num_pairs", "far", "frr"])
        writer.writeheader()
        writer.writerows(group_metrics)

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.axvline(summary["eer_threshold"], linestyle="--", label=f"EER threshold={summary['eer_threshold']:.3f}")
    plt.title("LFW: FAR/FRR vs Threshold")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(thresholds, fars, label="FAR")
    plt.plot(thresholds, frrs, label="FRR")
    plt.axvline(summary["eer_threshold"], linestyle="--", label=f"EER threshold={summary['eer_threshold']:.3f}")
    plt.title("LFW: FAR/FRR vs Threshold (Zoomed)")
    plt.xlabel("Threshold")
    plt.ylabel("Rate")
    plt.ylim(0.0, 0.2)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "far_frr_curve_zoom.png", dpi=150)
    plt.close()


def main():
    args = parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        device = "cpu"

    dataset = fetch_lfw_pairs(subset=args.subset, color=True, resize=1.0)
    pairs = dataset.pairs
    labels = dataset.target.astype(int)

    total_pairs = len(labels)
    use_pairs = min(args.max_pairs, total_pairs)

    embedder = FaceEmbedder(device=device)
    pipeline = FacePipeline(detector=None, embedder=embedder)

    scores = []
    eval_labels = []

    for index in range(use_pairs):
        image1 = to_pil(pairs[index, 0])
        image2 = to_pil(pairs[index, 1])

        emb1 = pipeline.get_embedding(image1)
        emb2 = pipeline.get_embedding(image2)

        score = pipeline.cosine_similarity(emb1, emb2)
        scores.append(score)
        eval_labels.append(int(labels[index]))

        if (index + 1) % 100 == 0:
            print(f"Processed {index + 1}/{use_pairs} pairs...")

    thresholds, fars, frrs, eer_threshold, eer = compute_curves(
        scores,
        eval_labels,
        args.threshold_min,
        args.threshold_max,
        args.threshold_step,
    )

    decision_threshold = args.decision_threshold if args.decision_threshold is not None else eer_threshold

    groups = build_proxy_groups(use_pairs)
    group_metrics = compute_group_metrics(scores, eval_labels, groups, decision_threshold)

    score_stats = {
        "score_min": float(np.min(scores)),
        "score_max": float(np.max(scores)),
        "score_p05": float(np.percentile(scores, 5)),
        "score_p50": float(np.percentile(scores, 50)),
        "score_p95": float(np.percentile(scores, 95)),
    }

    summary = {
        "dataset": "LFW",
        "subset": args.subset,
        "total_pairs_available": int(total_pairs),
        "evaluated_pairs": int(use_pairs),
        "eer": float(eer),
        "eer_threshold": float(eer_threshold),
        "decision_threshold": float(decision_threshold),
        "fairness_note": "LFW loader lacks demographic metadata in this pipeline; group metrics use proxy buckets.",
        **score_stats,
    }

    output_dir = Path(args.output_dir)
    save_outputs(output_dir, thresholds, fars, frrs, summary, group_metrics)

    print("Evaluation complete")
    print(json.dumps(summary, indent=2))
    print(f"Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
