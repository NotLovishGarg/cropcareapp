from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import tempfile
import os

app = Flask(__name__)

model = tf.keras.models.load_model("trained_plant_disease_model.keras")

class_labels = [
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy',
    'Potato___Early_blight',
    'Potato___Late_blight',
    'Potato___healthy',
    'Tomato___Bacterial_spot',
    'Tomato___Early_blight',
    'Tomato___Late_blight',
    'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

def model_prediction(image_file):
    try:
        
        image = tf.keras.preprocessing.image.load_img(image_file,target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) 
        predictions = model.predict(input_arr)
        return np.argmax(predictions) 
    except Exception as e:
        print(f"Error processing image: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/login')
def login():
    return render_template('ll.html')

@app.route('/signup')
def signup():
    return render_template('sign.html')

@app.route('/Tomato___Early_blight')
def teblight():
    return render_template('Tomato___Early_blight.html')

@app.route('/Tomato___Late_blight')
def tlblight():
    return render_template('Tomato___Late_blight.html')

@app.route('/Tomato___Bacterial_spot')
def tbspot():
    return render_template('Tomato___Bacterial_spot.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        
        image_file = request.files['image']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_file.read())
            temp_file_path = temp_file.name

        
        result_index = model_prediction(temp_file_path)

        os.remove(temp_file_path)
        
        if result_index is None:
            return jsonify({'error': 'Error processing the image'}), 500

        
        predicted_disease = class_labels[result_index]

        
        return jsonify({
            'disease': predicted_disease
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500


