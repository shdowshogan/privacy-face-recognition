# This script is for demonstration only. It uses two provided images for face detection and embedding comparison.
from PIL import Image
from detector import FaceDetector
from embedder import FaceEmbedder
from pipeline import FacePipeline

# Image filenames (ensure these are saved in the same directory as this script)
IMG1 = 'image1.jpg'
IMG2 = 'image2.jpg'
THRESHOLD = 0.9

def main():
    detector = FaceDetector()
    embedder = FaceEmbedder()
    pipeline = FacePipeline(detector, embedder)

    img1 = Image.open(IMG1).convert('RGB')
    img2 = Image.open(IMG2).convert('RGB')

    face1 = pipeline.detect_face(img1)
    face2 = pipeline.detect_face(img2)

    if face1 is None or face2 is None:
        print('Face not detected in one or both images.')
        return

    emb1 = pipeline.get_embedding(face1)
    emb2 = pipeline.get_embedding(face2)

    similarity = pipeline.cosine_similarity(emb1, emb2)
    verified = similarity >= THRESHOLD
    print(f'Cosine similarity between the two images: {similarity:.4f}')
    print(f'Threshold: {THRESHOLD:.2f}')
    print(f'Verified: {verified}')

if __name__ == '__main__':
    main()
