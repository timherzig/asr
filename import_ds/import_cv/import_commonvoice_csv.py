import os
import wave
import librosa
import warnings
import numpy as np
import pandas as pd

def speech_file_to_array_hf(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)

    return speech_array

def speech_file_to_array_ds(x):

    w = wave.open(x, 'r')
    frames = w.getnframes()
    buffer = w.readframes(frames)
    speech_array = np.frombuffer(buffer, dtype=np.int16)

    return speech_array

def import_commonvoice_csv(path, toolkit):
    location, file = os.path.split(path)
    print('Importing {} from {}'.format(file, location))

    df = pd.read_csv(path)
    df = df.sample(n=500) # Remove if the whole df is to be used

    if toolkit == 'huggingface':
        df['audio'] = [speech_file_to_array_hf(os.path.join(location, x)) for x in df['wav_filename']]
    elif toolkit == 'deepspeech':
        df['audio'] = [speech_file_to_array_ds(os.path.join(location, x)) for x in df['wav_filename']]

    df = df.drop(columns=['wav_filename'])
    df = df.drop(columns=['wav_filesize'])

    return df