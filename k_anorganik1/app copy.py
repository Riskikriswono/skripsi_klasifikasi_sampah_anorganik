from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = 'best_model_densenet121_501d_643.keras'  
model = load_model(model_path)

# Dictionary for mapping predictions
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic', 'Empty']

def predict_label(img_path):
    i = image.load_img(img_path, target_size=(224, 224))
    i = image.img_to_array(i) / 255.0
    i = np.expand_dims(i, axis=0)
    p = model.predict(i)
    index = np.argmax(p)
    return labels[index]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            img_path = os.path.join('static', file.filename)
            file.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', prediction=prediction, img_path=img_path)
    
    return render_template('index.html')

@app.route('/model', methods=['GET', 'POST'])
def model():
    return render_template('model.html')

if __name__ == '__main__':
    app.run(debug=True)
