# Phase 2 Evaluation (FAR / FRR / EER)

This folder contains a runnable evaluator for the face verification pipeline.

## Input CSV format

Provide a CSV with:

- `img1`: absolute or relative path to first image
- `img2`: absolute or relative path to second image
- `label`: `1` for same person, `0` for different person
- `group` (optional): demographic group key for fairness reporting (gender/age/skin-tone bucket)

See `pairs_template.csv` for an example.

## Run

From repository root:

```powershell
.\.venv312-py312\Scripts\python.exe .\face_pipeline\evaluation\evaluate_pairs.py --pairs-csv .\face_pipeline\evaluation\pairs_template.csv --output-dir .\face_pipeline\evaluation\out
```

## Outputs

- `metrics_summary.json` - EER + selected threshold summary
- `threshold_metrics.csv` - FAR/FRR values across thresholds
- `group_metrics.csv` - FAR/FRR per group at decision threshold
- `far_frr_curve.png` - FAR/FRR plot
- `far_frr_curve_zoom.png` - zoomed FAR/FRR plot (0 to 0.2 y-range)
- `skipped_pairs.csv` - rows skipped (missing files/face detection failure)

### Why the full curve can look flat

When threshold is swept across the full cosine range (`-1` to `1`), FAR/FRR can appear nearly horizontal for long regions, especially when the model separates pairs well. Use `far_frr_curve_zoom.png` for operating-region interpretation.

## Notes

- Evaluator only persists numerical outputs and logs, not raw face images.
- For Phase 2 deliverables, run on LFW or VGGFace2 subset and include the output plot in your main README.
