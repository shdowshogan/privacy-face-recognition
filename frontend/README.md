# Phase 5 Frontend (React)

This frontend consumes the FastAPI backend and includes:

- Enrollment flow
- Verification flow
- Consent management
- Metrics dashboard from `/metrics`

## Run

Start backend first (from repo root):

```powershell
.\.venv312-py312\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

In a second terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open:

- `http://127.0.0.1:5173`

## Notes

- Default API base URL in UI is `http://127.0.0.1:8000`.
- Backend CORS is enabled for `localhost:5173` and `127.0.0.1:5173`.
