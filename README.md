# Privacy-Preserving Face Recognition System

## Ethical Design Choices

- **No raw face images stored**: The system never persists raw face images at any stage.
- **Only face embeddings persisted**: Only the mathematical representation (embedding) of a face is stored, not the image itself.
- **User explicit consent required**: Users must provide explicit consent before enrollment. No hidden or automatic enrollment.
- **Revocation deletes embeddings immediately**: If a user revokes consent, all their embeddings are hard deleted from the system with no soft delete or backup.

### Explicit Non-Goals
- Not for surveillance
- Not for mass identification
- No hidden enrollment

These principles are foundational and enforced throughout the system design and implementation.

## GitHub Pages Deployment (Frontend)

This project supports GitHub Pages deployment for the React frontend.

Important: GitHub Pages is static hosting only. The FastAPI backend must be hosted separately (Render/Railway/Fly.io/etc.).

### What is configured

- Vite production base path for this repo: `/privacy-face-recognition/`
- GitHub Actions workflow: `.github/workflows/deploy-frontend-pages.yml`
- Frontend API URL via `VITE_API_BASE` variable

### One-time GitHub setup

1. Push latest code to `main`.
2. In GitHub repo settings, go to **Settings → Pages** and set **Build and deployment** source to **GitHub Actions**.
3. In **Settings → Secrets and variables → Actions → Variables**, add:
	- `VITE_API_BASE` = your public backend URL (for example, `https://your-backend-domain.com`).

After this, each push affecting `frontend/**` triggers deployment automatically.

## Next Step (Phase 2: Evaluation)

The immediate next milestone is metrics and fairness evaluation.

1. Build a labeled pairs CSV (`img1,img2,label,group`) using LFW or a VGGFace2 subset.
2. Run the evaluator to compute FAR/FRR curves and EER.
3. Include the generated FAR/FRR plot and summary metrics in this README.

Run:

```powershell
.\.venv312-py312\Scripts\python.exe .\face_pipeline\evaluation\evaluate_pairs.py --pairs-csv .\face_pipeline\evaluation\pairs_template.csv --output-dir .\face_pipeline\evaluation\out
```

## Phase 2 Benchmark Results (LFW)

Run configuration:

- Dataset: LFW (`10_folds`)
- Evaluated pairs: 1000 (from 6000 available)
- Output folder: `face_pipeline/evaluation/lfw_out`

Observed metrics:

- EER: 0.0083
- EER threshold: 0.37
- Decision threshold: 0.37

Group-wise FAR/FRR (current run):

- `proxy_bucket_0`: FAR 0.01, FRR 0.00
- `proxy_bucket_1`: FAR 0.01, FRR 0.0133
- `proxy_bucket_2`: FAR 0.01, FRR 0.0067
- `proxy_bucket_3`: FAR 0.01, FRR 0.0067

Fairness note:

- LFW in this pipeline does not include demographic attributes (gender/skin tone/age) in the evaluator path.
- Group-wise values above are proxy buckets to validate group-metrics plumbing.
- For a real fairness audit, attach demographic metadata and compute FAR/FRR per demographic group.

Generated artifacts:

- `face_pipeline/evaluation/lfw_out/far_frr_curve.png`
- `face_pipeline/evaluation/lfw_out/metrics_summary.json`
- `face_pipeline/evaluation/lfw_out/group_metrics.csv`
