import argparse
from pathlib import Path

from PIL import Image

from detector import FaceDetector
from embedder import FaceEmbedder
from pipeline import FacePipeline
from privacy_store import PrivacyStore


def parse_args():
    parser = argparse.ArgumentParser(description="Enroll a user face embedding (no image storage).")
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--image", required=True, help="Path to enrollment face image")
    parser.add_argument("--consent", required=True, choices=["yes", "no"])
    parser.add_argument("--db", default="face_pipeline/privacy.db")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    consent = args.consent.lower() == "yes"

    if not consent:
        raise ValueError("Enrollment blocked: explicit consent must be 'yes'.")

    image_path = Path(args.image)
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    detector = FaceDetector(device=args.device)
    embedder = FaceEmbedder(device=args.device)
    pipeline = FacePipeline(detector, embedder)

    image = Image.open(image_path).convert("RGB")
    face = pipeline.detect_face(image)
    if face is None:
        raise RuntimeError("No face detected in enrollment image.")

    embedding = pipeline.get_embedding(face)

    store = PrivacyStore(db_path=args.db)
    store.enroll_embedding(args.user_id, embedding, consent=True)

    print(f"Enrolled user: {args.user_id}")
    print("Raw image not stored; only normalized embedding persisted.")


if __name__ == "__main__":
    main()
