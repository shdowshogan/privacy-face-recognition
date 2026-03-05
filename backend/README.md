# Phase 4 Backend (FastAPI)

Implemented endpoints:

- `POST /enroll`
- `POST /verify`
- `GET /users/{id}`
- `POST /users/{id}/revoke`
- `DELETE /users/{id}`
- `GET /metrics`

## Run API

From repository root:

```powershell
.\.venv312-py312\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

## Deploy (Render)

This repo includes:

- `render.yaml` (service definition)
- `requirements.txt` (Python dependencies)
- `runtime.txt` (Python runtime)

Render start command:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

Health check path:

- `/health`

Open docs:

- `http://127.0.0.1:8000/docs`

## Example requests (PowerShell)

Enroll:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/enroll" -F "user_id=user_1" -F "consent=true" -F "image=@image1.jpg"
```

Verify:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/verify" -F "threshold=0.37" -F "image=@image1.jpg"
```

Get user:

```powershell
curl.exe "http://127.0.0.1:8000/users/user_1"
```

Delete user (hard delete):

```powershell
curl.exe -X DELETE "http://127.0.0.1:8000/users/user_1"
```

Revoke consent (hard delete embeddings, keep user record with consent false):

```powershell
curl.exe -X POST "http://127.0.0.1:8000/users/user_1/revoke"
```

Get metrics:

```powershell
curl.exe "http://127.0.0.1:8000/metrics"
```
