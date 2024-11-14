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

Training:
Train the model on the Flickr dataset for desired number of epochs(Took around 4 hrs of training on NVIDIA 4070 Ti 12G for 5 epochs with lr=1e-4)

Running Web Application:
Once the training is completed, check whether the caption is being generated or not by running the main.py file. Run the app.py file and go to localhost server to get the web interface for image captioning. 

Try it yourself:
1. git clone https://github.com/git-khandelwal/CNN-to-GPT2
2. pip install -r requirements.txt
3. Download the pt weights: https://drive.google.com/file/d/1uANPY6WZusGcFPumj-jZgl3UP0IhLW9-/view?usp=sharing
4. flask run --host=0.0.0.0 --port=5000
5. Go to the IP address and upload images(.jpg/.png formats supported) to generate captions

Web Interface:
![image](https://github.com/user-attachments/assets/aea7b0fb-2f93-46ca-8ec8-d78c357d9eac)
