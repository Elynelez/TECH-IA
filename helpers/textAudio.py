import speech_recognition as sr
from pydub import AudioSegment

def convert_audio_to_wav(audio_file):
    # Cargar el archivo de audio usando pydub
    audio = AudioSegment.from_file(audio_file)
    wav_file_path = audio_file.rsplit('.', 1)[0] + '.wav'  # Cambiar extensión a .wav
    audio.export(wav_file_path, format='wav')  # Exportar como .wav
    return wav_file_path

def message(audio_file):
    recognizer = sr.Recognizer()

    # Convertir el archivo de audio a WAV
    wav_file_path = convert_audio_to_wav(audio_file)

    # Usar el archivo WAV para el reconocimiento
    with sr.AudioFile(wav_file_path) as source:
        try:
            # Ajustar para el ruido ambiental
            recognizer.adjust_for_ambient_noise(source, duration=0.2)
            
            # Escuchar el audio del archivo
            audio = recognizer.record(source)  # Graba el audio completo

            # Reconocer el habla usando el servicio de Google
            text = recognizer.recognize_google(audio, language="es-CO")  
            text = text.lower()

            # Retornar el texto reconocido en formato JSON
            return {'message': text}

        except sr.UnknownValueError:
            return {'message': "No se pudo entender el audio."}

        except sr.RequestError:
            return {'message': "Error en el servicio de reconocimiento de Google."}