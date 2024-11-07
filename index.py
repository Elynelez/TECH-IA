import sys
import json
import scipy.signal as signal
import soundfile as sf
import os
import numpy as np
import librosa
import noisereduce as nr
from pydub import AudioSegment
from helpers.cutAudios import cut_and_normalize_audios
from helpers.textAudio import message
from helpers.modelAudios import extract_features, create_model, load_saved_model

# Probar un archivo de audio nuevo
def test_audio(audio_file, model, label_encoder):
    # Extraer características del archivo de audio de prueba
    feature = extract_features(audio_file).reshape(1, -1)
    
    # Predecir probabilidades para cada clase
    class_probabilities = model.predict_proba(feature).flatten()
    
    # Obtener el índice de la clase predicha
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

if __name__ == "__main__":
    # Asegurarse de que se pase el archivo de audio como argumento
    if len(sys.argv) > 1:

        # Entrenar el modelo
        if os.path.exists('model.pkl') and os.path.exists('label_encoder.pkl'):
            clf, le = load_saved_model()
        else:
            clf, le = create_model()

        # # Probar un audio de prueba
        audio_file_path = sys.argv[1]
        enhanced_audio_file = enhance_audio(audio_file_path)
        class_probabilities, predicted_class_index = test_audio(enhanced_audio_file, clf, le)
        
        # Mostrar las probabilidades por clase
        classes = le.classes_
        
        # # Mostrar la clase y precisión predichas
        predicted_class = classes[predicted_class_index]
        accuracy = class_probabilities[predicted_class_index]

        message_response = message(audio_file_path)
        
        message_response["predicted_class"] = predicted_class
        message_response["accuracy"] = round(accuracy, 4)

        print(json.dumps(message_response))
    else:
        print(json.dumps({'message': "No se proporcionó un archivo de audio."}))
    
    

    