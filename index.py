from flask import Flask, request, jsonify
from flask_cors import CORS
import librosa
import numpy as np
from tensorflow.keras.models import load_model
import noisereduce as nr
from pydub import AudioSegment
import scipy.signal as signal
import soundfile as sf
import os
from helpers.textAudio import message
from helpers.models.svcModel import extract_features, load_saved_model

app = Flask(__name__)
CORS(app, origins=["http://localhost:3000"])

model = load_model('audio_classification_model.h5')
classes = ['Diego', 'Edison', 'Elian', 'Nadie', 'Eddy']
fixed_length = 50

def compute_mfccs(audio, n_mfcc=40, fixed_length=156, window_size=1024, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length)
    current_length = mfccs.shape[1]
    return np.pad(mfccs, ((0, 0), (0, fixed_length - current_length)), mode='constant') if current_length < fixed_length else mfccs[:, :fixed_length]

def test_audio_mfccs(file_path, model):
    audio_data, sample_rate = librosa.load(file_path, sr=16000)
    mfccs = compute_mfccs(audio_data)
    mfccs = np.expand_dims(mfccs, axis=0)
    mfccs = np.expand_dims(mfccs, axis=-1)
    predictions = model.predict(mfccs)
    class_probabilities = predictions[0]
    predicted_class_index = np.argmax(class_probabilities)
    return class_probabilities, predicted_class_index

def enhance_audio(audio_file):
    # Normalización de audio
    def normalize_audio(audio):
        return audio.apply_gain(-audio.max_dBFS)

    # Cargar el archivo de audio
    audio = AudioSegment.from_file(audio_file)
    
    # Normalizar el audio
    audio_normalized = normalize_audio(audio)
    
    # Guardar el audio normalizado temporalmente
    normalized_path = "temp_normalized.wav"
    audio_normalized.export(normalized_path, format="wav")
    
    # Cargar el audio normalizado para reducción de ruido
    y, sr = librosa.load(normalized_path, sr=None)
    
    # Aplicar reducción de ruido
    y_denoised = nr.reduce_noise(y=y, sr=sr)
    
    # Filtrado de paso alto
    def highpass_filter(audio, sr, cutoff=300):
        sos = signal.butter(10, cutoff, 'hp', fs=sr, output='sos')
        return signal.sosfilt(sos, audio)

    y_filtered = highpass_filter(y_denoised, sr)
    
    # Guardar el audio mejorado
    enhanced_audio_path = os.path.splitext(audio_file)[0] + "_enhanced.wav"
    sf.write(enhanced_audio_path, y_filtered, sr)
    
    # Eliminar el archivo temporal
    os.remove(normalized_path)
    
    return enhanced_audio_path

def test_audio_svc(audio_file, model, label_encoder):
    # Extraer características del archivo de audio de prueba
    feature = extract_features(audio_file).reshape(1, -1)
    
    # Predecir probabilidades para cada clase
    class_probabilities = model.predict_proba(feature).flatten()
    
    # Obtener el índice de la clase predicha
    predicted_class_index = np.argmax(class_probabilities)
    
    return class_probabilities, predicted_class_index

@app.route('/predict/tfmodel', methods=['POST'])
def predict_tfmodel():
    # Obtener los datos JSON del cuerpo de la solicitud
    data = request.get_json()

    # # Acceder a 'audioFilePath' desde el JSON
    file_path = data.get('audioFilePath')
    enhanced_audio_file = enhance_audio(file_path)
    # Procesa la solicitud y obtiene las probabilidades y el índice de clase predicha
    class_probabilities, predicted_class_index = test_audio_mfccs(enhanced_audio_file, model)
    predicted_class = classes[predicted_class_index]
    accuracy = float(class_probabilities[predicted_class_index])  # Conversión a float

    # Convierte los valores a tipos nativos de Python
    class_probabilities = [float(prob) for prob in class_probabilities]

    message_response = message(file_path)
        
    message_response["predicted_class"] = predicted_class
    message_response["accuracy"] = round(accuracy, 4)
    message_response['probabilities'] = {classes[i]: round(class_probabilities[i], 4) for i in range(len(classes))}

    return jsonify(message_response)

@app.route('/predict/svcmodel', methods=['POST'])
def predict_svcmodel():
    # Obtener los datos JSON del cuerpo de la solicitud
    data = request.get_json()

    # cargar modelos
    clf, le = load_saved_model()

    # # Probar un audio de prueba
    audio_file_path = data.get('audioFilePath')
    enhanced_audio_file = enhance_audio(audio_file_path)
    class_probabilities, predicted_class_index = test_audio_svc(enhanced_audio_file, clf, le)
    
    # Mostrar las probabilidades por clase
    classes = le.classes_
    
    # # Mostrar la clase y precisión predichas
    predicted_class = classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]

    message_response = message(audio_file_path)
    
    message_response["predicted_class"] = predicted_class
    message_response["accuracy"] = round(accuracy, 4)

    return jsonify(message_response)

@app.route('/popo', methods=['GET'])
def popo():
    return jsonify({'message': '23123'})

if __name__ == '__main__':
    app.run(host='localhost', debug=True, port=5000)    

    