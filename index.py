import sys
import json
from helpers.cutAudios import cut_and_normalize_audios
from helpers.textAudio import message

if __name__ == "__main__":
    cut_and_normalize_audios()
    # Asegurarse de que se pase el archivo de audio como argumento
    if len(sys.argv) > 1:
        audio_file_path = sys.argv[1]
        print(json.dumps(message(audio_file_path)))
    else:
        print(json.dumps({'message': "No se proporcionó un archivo de audio."}))