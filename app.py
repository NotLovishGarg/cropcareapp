from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
from PIL import Image
import tempfile
import os

app = Flask(__name__)

# Load the model
model = tf.keras.models.load_model("trained_plant_disease_model.keras")

# Define class labels
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
        # Open the image file using PIL
        image = tf.keras.preprocessing.image.load_img(image_file,target_size=(128,128))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.array([input_arr]) #convert single image to batch
        predictions = model.predict(input_arr)
        return np.argmax(predictions) #return index of max element
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

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    try:
        # Get the image file from the request
        image_file = request.files['image']

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
            temp_file.write(image_file.read())
            temp_file_path = temp_file.name

        # Get the prediction
        result_index = model_prediction(temp_file_path)

        os.remove(temp_file_path)
        
        if result_index is None:
            return jsonify({'error': 'Error processing the image'}), 500

        # Get the predicted disease from the class labels
        predicted_disease = class_labels[result_index]

        # Return the prediction as a JSON response
        return jsonify({
            'disease': predicted_disease
        })
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'Error processing image'}), 500

if __name__ == '__main__':
    app.run(debug=True)
