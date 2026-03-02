# Face embedding model using InceptionResnetV1 (facenet-pytorch)
from facenet_pytorch import InceptionResnetV1
import torch
from torchvision import transforms
from PIL import Image
import numpy as np

class FaceEmbedder:
    def __init__(self, device='cpu'):
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
        self.device = device
        self.pil_transform = transforms.Compose(
            [
                transforms.Resize((160, 160)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

    def embed(self, face_img):
        """
        Generate embedding from a face image tensor or PIL image.
        Returns a 512-d numpy array.
        """
        if torch.is_tensor(face_img):
            img = face_img
            if img.ndim == 3:
                img = img.unsqueeze(0)
            img = img.to(self.device)
        elif isinstance(face_img, Image.Image):
            img = self.pil_transform(face_img).unsqueeze(0)
            img = img.to(self.device)
        else:
            raise ValueError('Input must be a torch.Tensor or PIL Image')

        with torch.no_grad():
            emb = self.model(img)
        return emb.squeeze().cpu().numpy()
