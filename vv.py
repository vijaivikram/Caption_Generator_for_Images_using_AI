from flask import Flask, render_template, request
from transformers import pipeline
import os
from PIL import Image

app = Flask(__name__)

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def caption_generator():
    imagefile = request.files["imagefile"]
    imagepath = "./images/" + imagefile.filename
    imagefile.save(imagepath)
    with Image.open(imagefile) as img:
    
        image_captioner = pipeline("image-to-text", model="Abdou/vit-swin-base-224-gpt2-image-captioning")
        # infer the caption
        caption = image_captioner(img)[0]['generated_text']
    
    
    
        return render_template("index.html", prediction=caption)

if __name__=='__main__':
    app.run(debug=True)