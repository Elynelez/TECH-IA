from pydub import AudioSegment
import os
import numpy as np
import librosa
import noisereduce as nr
import scipy.signal as signal
import soundfile as sf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Función para extraer características con reducción de ruido
def extract_features(audio_file):
    y, sr = librosa.load(audio_file, sr=None)
    
    # Reducción de ruido
    y = nr.reduce_noise(y=y, sr=sr)
    
    # Normalización
    y = librosa.util.normalize(y)
    
    # Extraer características
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
    
    # Concatenar todas las características en un solo vector
    features = np.hstack([
        np.mean(mfccs, axis=1),
        np.mean(chroma_stft, axis=1),
        np.mean(spectral_contrast, axis=1)
    ])
    
    return features

# Cargar datos y extraer características
def load_data(data_dir):
    features = []
    labels = []
    
    for speaker in os.listdir(data_dir):
        speaker_dir = os.path.join(data_dir, speaker, 'cut')
        if os.path.isdir(speaker_dir):
            for audio_file in os.listdir(speaker_dir):
                if audio_file.endswith('.wav'):
                    file_path = os.path.join(speaker_dir, audio_file)
                    feature = extract_features(file_path)
                    features.append(feature)
                    labels.append(speaker)

    return np.array(features), np.array(labels)

def create_model():
    data_dir = 'dataset'
    features, labels = load_data(data_dir)

    # Codificar las etiquetas
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    # Dividir el conjunto de datos
    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    # Entrenar el clasificador
    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    # Predecir en el conjunto de prueba
    y_pred = clf.predict(X_test)

    # Imprimir el informe de clasificación
    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    return clf, le

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

if __name__ == '__main__':
    # Segmentar, normalizar y reducir ruido en audios originales
    cut_and_normalize_audios()
    
    # Entrenar el modelo
    clf, le = main()

    # Probar un audio de prueba
    test_audio_file = 'test5.wav'
    enhanced_audio_file = enhance_audio(test_audio_file)
    class_probabilities, predicted_class_index = test_audio(enhanced_audio_file, clf, le)
    
    # Mostrar las probabilidades por clase
    classes = le.classes_
    for class_name, probability in zip(classes, class_probabilities):
        print(f'Class: {class_name}, Probability: {probability:.4f}')
    
    # Mostrar la clase y precisión predichas
    predicted_class = classes[predicted_class_index]
    accuracy = class_probabilities[predicted_class_index]
    
    print(f'The audio is classified as: {predicted_class}')
    print(f'Accuracy: {accuracy:.4f}')