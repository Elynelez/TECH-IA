import os
from pydub import AudioSegment

def cut_and_normalize_audios():
    
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
                    if len(audio) < segment_length:
                        print(f'Archivo {audio_file} es demasiado corto para segmentar.')
                        continue
                    segment.export(os.path.join(cut_path, f"{audio_file}_part{i}.wav"), format="wav")

cut_and_normalize_audios()