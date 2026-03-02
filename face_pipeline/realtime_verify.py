import argparse

import cv2
from PIL import Image

from detector import FaceDetector
from embedder import FaceEmbedder
from pipeline import FacePipeline
from privacy_store import PrivacyStore


def parse_args():
    parser = argparse.ArgumentParser(description="Realtime webcam face verification against local embedding DB.")
    parser.add_argument("--threshold", type=float, default=0.37)
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--db", default="face_pipeline/privacy.db")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    detector = FaceDetector(device=args.device)
    embedder = FaceEmbedder(device=args.device)
    pipeline = FacePipeline(detector, embedder)
    store = PrivacyStore(db_path=args.db)

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("Press 'q' to quit.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)

        face, box = detector.detect_with_box(pil_image)
        status_text = "No face"
        color = (0, 0, 255)

        if face is not None:
            probe_embedding = pipeline.get_embedding(face)
            records = store.list_active_embeddings()

            best_user = None
            best_score = -1.0

            for user_id, enrolled_embedding in records:
                score = pipeline.cosine_similarity(probe_embedding, enrolled_embedding)
                if score > best_score:
                    best_score = score
                    best_user = user_id

            if best_user is not None and best_score >= args.threshold:
                status_text = f"Verified: {best_user} ({best_score:.3f})"
                color = (0, 255, 0)
            else:
                status_text = f"Unknown ({best_score:.3f})"
                color = (0, 165, 255)

            if box is not None:
                x1, y1, x2, y2 = [int(v) for v in box]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, status_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.putText(
            frame,
            f"threshold={args.threshold:.2f}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        cv2.imshow("Privacy Face Verify (Realtime)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
