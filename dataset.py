from torch.utils.data import Dataset, DataLoader
import pandas as pd
from PIL import Image
import os
from transformers import GPT2Tokenizer
import torch
import cv2
from torchvision import transforms
import numpy as np
from torch.nn.utils.rnn import pad_sequence


class ImageCaption(Dataset):
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    bos_token = '<|startoftext|>'
    eos_token = '<|endoftext|>'
    pad_token = '<|pad|>'

    # Add special tokens to the tokenizer
    tokenizer.add_special_tokens({'bos_token': bos_token, 'eos_token': eos_token, 'pad_token': pad_token})

    # Update tokenizer configuration
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id


    def __init__(self, root, captions, transform=None):
        self.root = root
        self.df = pd.read_csv(captions)
        self.transform = transform
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        captions = '<|startoftext|>' + self.captions[index] + '<|endoftext|>'
        img_id = self.imgs[index]
        img_path = os.path.join(self.root, img_id)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        tokenized_captions = self.__class__.tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized_captions['input_ids'].squeeze(0)

        return torch.tensor(img), input_ids

class ImageCaption_(Dataset):
    def __init__(self, root, captions, tokenizer, transform=None):
        self.root = root
        self.df = pd.read_csv(captions)

        self.transform = transform

        # Get img, caption columns
        self.imgs = self.df["image"]
        self.captions = self.df["caption"]
        self.tokenizer = tokenizer 

    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        captions = self.captions[index]
        img_id = self.imgs[index]
        img_path = os.path.join(self.root, img_id)
        img = cv2.imread(img_path)

        if self.transform is not None:
            img = self.transform(img)
        
        
        # self.tokenizer.pad_token = self.tokenizer.eos_token 

        tokenized_captions = self.tokenizer(captions, padding=True, truncation=True, return_tensors='pt')
        input_ids = tokenized_captions['input_ids'].squeeze(0)

        return torch.tensor(img), input_ids


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

def collate_fn(batch):
    imgs, captions = zip(*batch)
    
    # Stack images into a single tensor
    imgs = torch.stack(imgs, dim=0)
    
    # Pad captions to the same length
    captions = pad_sequence(captions, batch_first=True, padding_value=50258)
    
    return imgs, captions


# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# bos_token = '<|startoftext|>'
# eos_token = '<|endoftext|>'
# pad_token = '<|pad|>'

# # Add special tokens to the tokenizer
# tokenizer.add_special_tokens({'bos_token': bos_token, 'eos_token': eos_token, 'pad_token': pad_token})

# bos_token_id = tokenizer.bos_token_id
# eos_token_id = tokenizer.eos_token_id
# pad_token_id = tokenizer.pad_token_id

# dataset = ImageCaption(root=r'C:\\Users\\khand\\Documents\\CNN to GPT2\\Flickr\\Images', captions=r'C:\\Users\\khand\\Documents\\CNN to GPT2\\Flickr\\captions.txt', transform=transform)
# dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

# for a,b in dataloader:
#     print(a, b)
#     break
