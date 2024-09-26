from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import cv2
from transform import transform
from generator import generate_caption
from model import ImageCaptioningModel
from transformers import GPT2Tokenizer
import torch
from dataset import ImageCaption
from flask import render_template
import sqlite3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

embed_size, vocab_size = 256, 50259
model = ImageCaptioningModel(embed_size, vocab_size).to(device=device)
model.load_state_dict(torch.load(r"C:\Users\khand\Documents\CNN to GPT2\state_dict.pth"))


def save_caption_to_db(image_name, caption):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()

    # Check if the image already exists in the history
    c.execute("SELECT captions FROM history WHERE image_name=?", (image_name,))
    row = c.fetchone()

    if row:
        existing_caption = row[0]
        # Update the history with a new caption
        updated_caption = f"{existing_caption}\n{caption}"
        c.execute("UPDATE history SET captions=? WHERE image_name=?", (updated_caption, image_name))
    else:
        # Insert new entry
        c.execute("INSERT INTO history (image_name, captions) VALUES (?, ?)", (image_name, caption))

    conn.commit()
    conn.close()

def get_existing_captions(image_name):
    conn = sqlite3.connect('history.db')
    c = conn.cursor()

    c.execute("SELECT captions FROM history WHERE image_name=?", (image_name,))
    row = c.fetchone()

    conn.close()
    return row[0] if row else None


app = Flask(__name__)
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        image = cv2.imread(filepath)
        image = transform(image)
        captions = generate_caption(model, image, ImageCaption.tokenizer)   
        existing_captions = get_existing_captions(file.filename)
        save_caption_to_db(file.filename, captions)    
        return jsonify({'captions': captions, 'existing_captions': existing_captions})
        


if __name__ == '__main__':
    app.run(debug=True)
