"""
Test script for face detection and embedding extraction using both PIL and OpenCV.
Usage:
    python test_face_pipeline.py <image1> [<image2>]
"""
import sys
from PIL import Image
import cv2
import numpy as np
from detector import FaceDetector
from embedder import FaceEmbedder
from pipeline import FacePipeline


def load_image_pil(path):
    return Image.open(path).convert('RGB')

def load_image_cv2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(img)

def main():
    if len(sys.argv) < 2:
        print('Usage: python test_face_pipeline.py <image1> [<image2>]')
        return

    detector = FaceDetector()
    embedder = FaceEmbedder()
    pipeline = FacePipeline(detector, embedder)

    # Test with PIL
    img1_pil = load_image_pil(sys.argv[1])
    face1 = pipeline.detect_face(img1_pil)
    if face1 is None:
        print('No face detected in image 1 (PIL)')
        return
    emb1 = pipeline.get_embedding(face1)
    print('Embedding 1 (PIL):', emb1[:5], '...')

    # Test with OpenCV
    img1_cv2 = load_image_cv2(sys.argv[1])
    face1_cv2 = pipeline.detect_face(img1_cv2)
    if face1_cv2 is None:
        print('No face detected in image 1 (OpenCV)')
        return
    emb1_cv2 = pipeline.get_embedding(face1_cv2)
    print('Embedding 1 (OpenCV):', emb1_cv2[:5], '...')

    # If a second image is provided, compare similarity
    if len(sys.argv) > 2:
        img2_pil = load_image_pil(sys.argv[2])
        face2 = pipeline.detect_face(img2_pil)
        if face2 is None:
            print('No face detected in image 2 (PIL)')
            return
        emb2 = pipeline.get_embedding(face2)
        sim = pipeline.cosine_similarity(emb1, emb2)
        print(f'Cosine similarity (PIL): {sim:.4f}')

        img2_cv2 = load_image_cv2(sys.argv[2])
        face2_cv2 = pipeline.detect_face(img2_cv2)
        if face2_cv2 is None:
            print('No face detected in image 2 (OpenCV)')
            return
        emb2_cv2 = pipeline.get_embedding(face2_cv2)
        sim_cv2 = pipeline.cosine_similarity(emb1_cv2, emb2_cv2)
        print(f'Cosine similarity (OpenCV): {sim_cv2:.4f}')

if __name__ == '__main__':
    main()
