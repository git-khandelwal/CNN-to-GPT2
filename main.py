import cv2
from transform import transform
from generator import generate_caption
from model import ImageCaptioningModel
from transformers import GPT2Tokenizer
import torch
from dataset import ImageCaption

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size, vocab_size = 256, 50259
model = ImageCaptioningModel(embed_size, vocab_size).to(device=device)
model.load_state_dict(torch.load(r"C:\Users\khand\Documents\CNN to GPT2\state_dict.pth"))

image = cv2.imread(r"C:\Users\khand\Documents\CNN to GPT2\Flickr\Images\3747543364_bf5b548527.jpg")  # Implement this function to load and preprocess the image
image = transform(image)
caption = generate_caption(model, image, ImageCaption.tokenizer)
print(caption)