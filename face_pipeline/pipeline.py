import numpy as np

class FacePipeline:
    """
    Privacy-preserving face embedding pipeline.
    - Detects faces
    - Generates embeddings
    - Normalizes embeddings
    - Computes cosine similarity
    """
    def __init__(self, detector, embedder):
        self.detector = detector
        self.embedder = embedder

    def detect_face(self, image):
        """Detect and align face from image. Returns cropped face."""
        return self.detector.detect(image)

    def get_embedding(self, face_img):
        """Generate and normalize embedding from face image."""
        emb = self.embedder.embed(face_img)
        return emb / np.linalg.norm(emb)

    @staticmethod
    def cosine_similarity(e1, e2):
        """Compute cosine similarity between two embeddings."""
        return float(np.dot(e1, e2) / (np.linalg.norm(e1) * np.linalg.norm(e2)))
