import json
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from PIL import Image

from face_pipeline.detector import FaceDetector
from face_pipeline.embedder import FaceEmbedder
from face_pipeline.pipeline import FacePipeline
from face_pipeline.privacy_store import PrivacyStore

app = FastAPI(title="Privacy-Preserving Face Recognition API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173",
        "https://shdowshogan.github.io",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DEFAULT_THRESHOLD = 0.37
LFW_METRICS_PATH = Path("face_pipeline/evaluation/lfw_out/metrics_summary.json")
FAVICON_PATH = Path("face-scan.png")
_PIPELINE = None


def get_pipeline(device: str = "cpu"):
    global _PIPELINE
    if _PIPELINE is None:
        detector = FaceDetector(device=device)
        embedder = FaceEmbedder(device=device)
        _PIPELINE = FacePipeline(detector, embedder)
    return _PIPELINE


def read_image_from_upload(upload: UploadFile):
    try:
        image = Image.open(upload.file).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image upload: {exc}") from exc
    return image


@app.get("/")
def root():
    return {"status": "ok", "service": "privacy-face-recognition-api"}


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    if FAVICON_PATH.exists():
        return FileResponse(FAVICON_PATH, media_type="image/png")
    return Response(status_code=204)


@app.post("/enroll")
def enroll(
    user_id: str = Form(...),
    consent: bool = Form(...),
    image: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    if not consent:
        raise HTTPException(status_code=400, detail="Explicit consent required for enrollment")

    pipeline = get_pipeline()
    store = PrivacyStore()

    pil_image = read_image_from_upload(image)
    face = pipeline.detect_face(pil_image)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in enrollment image")

    embedding = pipeline.get_embedding(face)
    store.enroll_embedding(user_id, embedding, consent=True)

    return {
        "user_id": user_id,
        "consent": True,
        "stored": "embedding_only",
        "threshold": threshold,
    }


@app.post("/verify")
def verify(
    image: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    pipeline = get_pipeline()
    store = PrivacyStore()

    pil_image = read_image_from_upload(image)
    face = pipeline.detect_face(pil_image)
    if face is None:
        raise HTTPException(status_code=400, detail="No face detected in verification image")

    probe_embedding = pipeline.get_embedding(face)
    candidates = store.list_active_embeddings()

    if not candidates:
        return {
            "verified": False,
            "confidence": 0.0,
            "matched_user_id": None,
            "threshold": threshold,
            "reason": "no_consented_embeddings",
        }

    best_user = None
    best_score = -1.0
    for user_id, enrolled_embedding in candidates:
        score = pipeline.cosine_similarity(probe_embedding, enrolled_embedding)
        if score > best_score:
            best_score = score
            best_user = user_id

    verified = bool(best_score >= threshold)
    return {
        "verified": verified,
        "confidence": float(best_score),
        "matched_user_id": best_user if verified else None,
        "threshold": float(threshold),
    }


@app.get("/users/{user_id}")
def get_user(user_id: str):
    store = PrivacyStore()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user


@app.post("/users/{user_id}/revoke")
def revoke_user(user_id: str):
    store = PrivacyStore()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    store.revoke_consent(user_id)
    return {"revoked": True, "user_id": user_id, "delete_mode": "hard_delete_embeddings"}


@app.delete("/users/{user_id}")
def delete_user(user_id: str):
    store = PrivacyStore()
    user = store.get_user(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")

    store.delete_user(user_id)
    return {"deleted": True, "user_id": user_id, "delete_mode": "hard_delete"}


@app.get("/metrics")
def metrics():
    store = PrivacyStore()
    store_metrics = store.get_metrics()

    evaluation_metrics = None
    if LFW_METRICS_PATH.exists():
        try:
            evaluation_metrics = json.loads(LFW_METRICS_PATH.read_text(encoding="utf-8"))
        except Exception:
            evaluation_metrics = None

    return {
        "store_metrics": store_metrics,
        "evaluation_metrics": evaluation_metrics,
    }
