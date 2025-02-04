from flask import Flask, render_template, request, redirect, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

# Load the model
model_path = 'best_model_densenet201_32303a.keras'

try:
    model = load_model(model_path)
    print("Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")

# Dictionary for mapping predictions
labels = ['Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']

def predict_label(img_path):
    try:
        i = image.load_img(img_path, target_size=(224, 224))
        i = image.img_to_array(i) / 255.0
        i = np.expand_dims(i, axis=0)
        p = model.predict(i)
        index = np.argmax(p)
        return labels[index]
    except Exception as e:
        print(f"Error in prediction: {e}")
        return "Prediction error"

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            img_path = os.path.join('static/d201', file.filename)
            file.save(img_path)
            prediction = predict_label(img_path)
            return render_template('index.html', prediction=prediction, img_path=img_path)
    
    return render_template('index.html')

# Fungsi untuk mendapatkan ringkasan model
def get_model_summary(model):
    summary_str = []
    model.summary(print_fn=lambda x: summary_str.append(x))  
    return summary_str

@app.route('/model', methods=['GET', 'POST'])
def model_view():
    summary = get_model_summary(model) 
    return render_template('model.html', summary=summary)

# @app.route('/model', methods=['GET', 'POST'])
# def model_view():
#     return render_template('model.html')

if __name__ == '__main__':
    app.run(port=4444)
