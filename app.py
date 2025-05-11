from flask import Flask, request, render_template
import numpy as np
import librosa
import tensorflow as tf
import os

from utils.audio_utils import extract_mfcc

app = Flask(__name__)

from tensorflow.keras.models import load_model
model = load_model('model/model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400

    file = request.files['file']
    if file.filename == '':
        return "No file selected", 400

    y, sr = librosa.load(file, sr=None)
    mfccs = extract_mfcc(y, sr)
    mfccs = np.expand_dims(mfccs, axis=0)

    prediction = model.predict(mfccs)
    predicted_label = np.argmax(prediction, axis=1)[0]

    return f'Predicted Label: {predicted_label}'

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)


