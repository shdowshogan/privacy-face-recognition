# Privacy-Preserving Face Recognition System

End-to-end privacy-first face recognition project with:

- Face enrollment and verification APIs (FastAPI)
- Webcam-based user registration and live unlock UI (React)
- Consent-aware storage with revocation and hard deletion
- Evaluation pipeline with FAR/FRR/EER reporting

## Local Setup (Recommended)

This section is the fastest way for someone else to run the project locally.

### Prerequisites

- **Python 3.12.x** (important: avoid Python 3.13 for this stack)
- **Node.js 18+** and npm
- Git

### 1) Clone

```powershell
git clone https://github.com/shdowshogan/privacy-face-recognition.git
cd privacy-face-recognition
```

### 2) Create and activate Python environment (Windows PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 3) Install backend dependencies

```powershell
pip install -r requirements.txt
```

### 4) Run backend API

From repo root:

```powershell
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

API endpoints/docs:

- Swagger UI: `http://127.0.0.1:8000/docs`
- Health check: `http://127.0.0.1:8000/health`

### 5) Run frontend

Open a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173`

The frontend defaults to `http://127.0.0.1:8000` as API base, so no extra config is needed for local development.

### 6) Local user flow to test

1. Step 1: Register face (camera capture or upload), provide user ID and consent.
2. Step 2: Click **Start Live Camera** and view live verification score/status.
3. Optional: Enable Admin Mode for consent management and metrics.

## Optional Local Add-ons

If you want to run the dataset evaluation scripts and desktop OpenCV realtime script, install:

```powershell
pip install scikit-learn matplotlib opencv-python
```

## Troubleshooting

- **`py -3.12` not found**: install Python 3.12 and ensure the Python launcher is installed.
- **Camera not starting in browser**: allow webcam permission for `http://127.0.0.1:5173`.
- **CORS/API errors**: ensure backend is running on port `8000` and frontend API base matches.
- **Slow first inference**: first model load can take longer; subsequent requests are faster.

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

## Backend Deployment (Render)

Backend deployment is configured for Render using `render.yaml` and `requirements.txt`.

### What is configured

- Render blueprint file: `render.yaml`
- Python runtime: `runtime.txt`
- Health endpoint: `GET /health`
- CORS origins configurable via `CORS_ALLOWED_ORIGINS`

### Deploy steps

1. In Render, click **New +** → **Blueprint**.
2. Connect this GitHub repo and select the branch (`main`).
3. Render reads `render.yaml` and creates web service `privacy-face-recognition-api`.
4. Wait for deploy to complete and copy the public backend URL.

### Connect deployed backend to GitHub Pages frontend

1. In GitHub repo settings: **Settings → Secrets and variables → Actions → Variables**.
2. Set/update:
	- `VITE_API_BASE` = your Render backend URL (for example, `https://privacy-face-recognition-api.onrender.com`).
3. Re-run or re-trigger the Pages workflow so frontend rebuilds with the new API base.

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
