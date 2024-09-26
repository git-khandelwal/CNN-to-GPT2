import torch
import torch.nn as nn
import torchvision.models as models
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        # To get the most recent weights for ResNet50
        # resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        for param in resnet.parameters():
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

class GPT2_Decoder(torch.nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(GPT2_Decoder, self).__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2')
        self.gpt2.resize_token_embeddings(vocab_size)
        self.linear = torch.nn.Linear(embed_size, self.gpt2.config.n_embd)

    def forward(self, features, captions):
        features = self.linear(features).unsqueeze(1)
        inputs_embeds = self.gpt2.transformer.wte(captions)
        inputs_embeds = torch.cat((features, inputs_embeds), dim=1)
        outputs = self.gpt2(inputs_embeds=inputs_embeds, return_dict=True)
        return outputs.logits

class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, vocab_size):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = GPT2_Decoder(embed_size, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

