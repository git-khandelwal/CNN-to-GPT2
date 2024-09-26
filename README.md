Image Captioning Using CNNs and Transformers

This project implements an image captioning model that generates descriptive captions for input images. It uses a ResNet-50 model for image feature extraction and GPT-2 for generating captions based on these features. The model is trained on the Flickr dataset and deployed to process images by uploading them via a web interface.

Project Overview
The image captioning system consists of two main components:

Image Encoder: ResNet-50 (pretrained on ImageNet) is used to extract feature vectors from images.
Caption Generator: GPT-2 (pretrained) generates natural language captions based on the image features extracted by ResNet-50.

Key Features
Pretrained Models: Both ResNet-50 and GPT-2 are pretrained on large datasets to leverage transfer learning.
Image-to-Text Translation: The model takes an image as input and outputs a descriptive caption.
Web Interface: Users can upload an image through a simple frontend, and the backend generates and displays captions. 
