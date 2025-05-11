from flask import Flask, render_template, request, jsonify
import librosa
import numpy as np
import tensorflow as tf
import os
from utils.audio_utils import extract_mfcc

app = Flask(__name__)
model = tf.keras.models.load_model("model/your_model.h5")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio_file']
    y, sr = librosa.load(file, sr=None)
    features = extract_mfcc(y, sr)
    features = features.reshape(1, -1, 1)

    prediction = model.predict(features)
    result = "Stutter Detected" if prediction[0][0] > 0.5 else "Fluent Speech"

    return jsonify({'result': result, 'confidence': float(prediction[0][0])})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Render provides this
    app.run(host='0.0.0.0', port=port)

