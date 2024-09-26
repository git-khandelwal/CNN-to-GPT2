import numpy as np
import torch
from torchvision import transforms


class OpenCVToTensor:
    def __call__(self, img):
        img = np.array(img)
        img = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        return img
    

transform = transforms.Compose([
    transforms.ToPILImage(),              # Convert OpenCV image to PIL image
    transforms.Resize((224, 224)),        # Resize the image
    OpenCVToTensor(),                     # Convert to Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


 