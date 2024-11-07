import os
import numpy as np
import librosa
import joblib
import noisereduce as nr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Función para extraer características con reducción de ruido
def extract_features(audio_file):
    try:
        y, sr = librosa.load(audio_file, sr=None)
        
        # Reducción de ruido
        y = nr.reduce_noise(y=y, sr=sr)
        
        # Normalización
        y = librosa.util.normalize(y)
        
        # Extraer características
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=512)

        
        # Concatenar todas las características
        features = np.hstack([
            np.mean(mfccs, axis=1),
            np.mean(chroma_stft, axis=1),
            np.mean(spectral_contrast, axis=1)
        ])
        
        return features
    except Exception as e:
        print(f"Error procesando {audio_file}: {e}")
        return None
    
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

    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)

    X_train, X_test, y_train, y_test = train_test_split(features, labels_encoded, test_size=0.2, random_state=42)

    clf = SVC(kernel='linear', probability=True)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    classification_report(y_test, y_pred, target_names=le.classes_)

    # Guardar el modelo y el codificador
    joblib.dump(clf, 'model.pkl')
    joblib.dump(le, 'label_encoder.pkl')
    
    return clf, le

def load_saved_model():
    # Cargar el modelo y el codificador
    clf = joblib.load('model.pkl')
    le = joblib.load('label_encoder.pkl')
    return clf, le