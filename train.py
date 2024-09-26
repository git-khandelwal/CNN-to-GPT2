import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch.optim as optim
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from dataset import ImageCaption
from PIL import Image
from torchvision import transforms
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from model import ImageCaptioningModel
from transform import transform
from dataset import collate_fn


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_size = 256
vocab_size = 50259
max_seq_length = 20
learning_rate = 1e-4
num_epochs = 5


model = ImageCaptioningModel(embed_size, vocab_size).to(device=device)
criterion = torch.nn.CrossEntropyLoss(ignore_index=ImageCaption.pad_token_id)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


dataset = ImageCaption(root='C:\\Users\\khand\\Documents\\CNN to GPT2\\Flickr\\Images', captions='C:\\Users\\khand\\Documents\\CNN to GPT2\\Flickr\\captions.txt', transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)


model.train()

for epoch in range(num_epochs):
    for images, captions in dataloader:
        images, captions = images.to(device), captions.to(device)
        outputs = model(images, captions)
        outputs = outputs[:, :-1, :].contiguous()
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')


torch.save(model.state_dict(), "state_dict.pth")