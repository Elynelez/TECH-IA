import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import random
import shutil
import seaborn as sns
import soundfile as sf
import tensorflow as tf
# import torch
import zipfile
# from google.colab import files
from IPython import display
from pydub import AudioSegment
from IPython.display import Audio, display
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm
import tensorflow as tf

segment_length = 5000  # Duración de los segmentos en milisegundos (5 segundos)
dataset_path = 'dataset'

for file_name in os.listdir(dataset_path):
    org_path = os.path.join(dataset_path, file_name, 'org')
    cut_path = os.path.join(dataset_path, file_name, 'cut')

    if os.path.exists(cut_path):
        continue
    
    # Crea la carpeta 'cut' si no existe
    os.makedirs(cut_path, exist_ok=True)
    
    # Procesa cada archivo en la carpeta 'org'
    for audio_file in os.listdir(org_path):
        if audio_file.endswith('.wav'):
            audio = AudioSegment.from_wav(os.path.join(org_path, audio_file))
            
            # Segmenta el audio en fragmentos de 5 segundos
            for i, start in enumerate(range(0, len(audio), segment_length)):
                segment = audio[start:start + segment_length]
                segment = segment.apply_gain(-segment.max_dBFS)  # Normalización
                segment.export(os.path.join(cut_path, f"{audio_file}_part{i}.wav"), format="wav")

#csv metadata
csv_path = 'csv'
data = []

for file_name in os.listdir(dataset_path):
    org_path = os.path.join(dataset_path, file_name, 'org')
    cut_path = os.path.join(dataset_path, file_name, 'cut')

    if os.path.isdir(cut_path):
        for filename in os.listdir(cut_path):
            if filename.endswith('.wav'):
                file_path = os.path.join(cut_path, filename)
                data.append({'file': file_path, 'target': file_name})

if data:
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(csv_path, 'metadata.csv'), index=False)

    print(f'Dataset creado y guardado en: {csv_path}')
else:
    print('No se encontraron archivos .wav en las carpetas de Dataset.')


def trim_silence(audio):
    return librosa.effects.trim(audio)[0]

def load_audio(filename):
    audio, sr = librosa.load(filename, sr=16000, duration=5)
    audio = audio / np.max(np.abs(audio))
    audio = trim_silence(audio)
    return audio, sr

def compute_mfccs(audio, n_mfcc=40, fixed_length=156, window_size=1024, hop_length=512):
    mfccs = librosa.feature.mfcc(y=audio, sr=16000, n_mfcc=n_mfcc, n_fft=window_size, hop_length=hop_length)
    current_length = mfccs.shape[1]
    return np.pad(mfccs, ((0, 0), (0, fixed_length - current_length)), mode='constant') if current_length < fixed_length else mfccs[:, :fixed_length]

def load_features(filename, fixed_length=156, n_mfcc=40, window_size=1024, hop_length=512):
    audio, sr = load_audio(filename)
    mfccs_padded = compute_mfccs(audio, n_mfcc, fixed_length, window_size, hop_length)
    return mfccs_padded

# Inicializa listas para almacenar las características y etiquetas
features_mfccs = []
targets = []

# Recorrer la estructura de carpetas
for persona in os.listdir(dataset_path):
    cut_path = os.path.join(dataset_path, persona, 'cut')
    if os.path.isdir(cut_path):
        for file in os.listdir(cut_path):
            if file.lower().endswith('.wav'):
                file_path = os.path.join(cut_path, file)

                try:
                    # Cargar y extraer características
                    mfccs = load_features(file_path)  # Asegúrate de tener esta función implementada

                    features_mfccs.append(mfccs)

                    # Agregar la etiqueta (nombre de la persona)
                    targets.append(persona)

                except ValueError as e:
                    print(f"Error al procesar {file_path}: {e}")

# Convertir las listas a arrays de numpy
X_mfccs = np.array(features_mfccs)

# Añadir una dimensión adicional a X_mfccs si es necesario
X_mfccs = np.expand_dims(X_mfccs, axis=-1)

# Convertir las etiquetas a array de numpy
y = np.array(targets)

print("Extracción de características completada.")

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Realizar el split
X_train, X_test, y_train, y_test = train_test_split(X_mfccs, y_encoded, test_size=0.2, random_state=42)

print(f"Training MFCC shape: {X_train.shape}, Training labels shape: {y_train.shape}")
print(f"Testing MFCC shape: {X_test.shape}, Testing labels shape: {y_test.shape}")

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from keras.models import Model

def create_model(num_classes):
    input_mfcc = Input(shape=(40, 156, 1))

    x_mfcc = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(input_mfcc)
    x_mfcc = BatchNormalization()(x_mfcc)
    x_mfcc = MaxPooling2D(pool_size=(2, 2))(x_mfcc)
    x_mfcc = Dropout(0.3)(x_mfcc)  # Dropout layer to prevent overfitting

    x_mfcc = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_mfcc)
    x_mfcc = BatchNormalization()(x_mfcc)
    x_mfcc = MaxPooling2D(pool_size=(2, 2))(x_mfcc)
    x_mfcc = Dropout(0.3)(x_mfcc)  # Dropout layer to prevent overfitting

    x_mfcc = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_mfcc)
    x_mfcc = BatchNormalization()(x_mfcc)
    x_mfcc = MaxPooling2D(pool_size=(2, 2))(x_mfcc)
    x_mfcc = Dropout(0.4)(x_mfcc)  # Dropout layer to prevent overfitting

    x_mfcc = Flatten()(x_mfcc)
    x_mfcc = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))(x_mfcc)
    x_mfcc = Dropout(0.5)(x_mfcc)  # Dropout layer before the final layer

    output = Dense(num_classes, activation='softmax')(x_mfcc)

    model = Model(inputs=input_mfcc, outputs=output)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

num_classes = 5
model = create_model(num_classes)

model.summary()

from tensorflow.keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    X_train,
    y_train,
    batch_size=32,
    epochs=3,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping]
)

# Evaluar el modelo
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy}")

# Hacer predicciones
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = y_test

# Imprimir reporte de clasificación
print(classification_report(y_true, y_pred_classes, target_names=label_encoder.classes_))

# Matriz de confusión
conf_matrix = confusion_matrix(y_true, y_pred_classes)

# Visualizar la matriz de confusión
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Save the model
model.save('audio_classification_model.h5')