# Face Embedding Pipeline

This module implements the privacy-preserving face embedding pipeline.

- Face detection (RetinaFace or MTCNN)
- Face embedding (ArcFace or FaceNet)
- Embedding normalization
- Cosine similarity computation
- No images are persisted

See README for ethical guarantees.

## Realtime Verification (Local Demo)

This project now supports a local realtime webcam verification flow:

1. Enroll user embedding (requires explicit consent)
2. Open webcam window
3. Detect face in realtime
4. Compare against stored embeddings
5. Display `Verified` or `Unknown`

### Enroll

```powershell
python .\face_pipeline\enroll_user.py --user-id user_1 --image .\image1.jpg --consent yes
```

### Revoke consent (hard delete embeddings)

```powershell
python .\face_pipeline\revoke_user.py --user-id user_1
```

### Realtime verify

```powershell
python .\face_pipeline\realtime_verify.py --threshold 0.37
```

Press `q` to close the webcam window.