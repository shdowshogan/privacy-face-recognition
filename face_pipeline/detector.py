# MTCNN face detector using facenet-pytorch
from facenet_pytorch import MTCNN

class FaceDetector:
    def __init__(self, device='cpu'):
        self.mtcnn = MTCNN(image_size=160, margin=0, device=device)

    def detect(self, image):
        """
        Detect and align face from an input image.
        Returns aligned face tensor (C, H, W) or None if not found.
        """
        return self.mtcnn(image)

    def detect_with_box(self, image):
        """
        Detect face and return aligned face tensor and first bounding box.
        Returns (face_tensor, box) where box is [x1, y1, x2, y2], or (None, None).
        """
        boxes, _ = self.mtcnn.detect(image)
        if boxes is None or len(boxes) == 0:
            return None, None
        face = self.mtcnn(image)
        if face is None:
            return None, None
        return face, boxes[0]
