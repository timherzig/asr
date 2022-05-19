import os
import librosa
import warnings
import pandas as pd

def speech_file_to_array(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)

    return speech_array

def import_commonvoice_csv(path):
    location, file = os.path.split(path)
    print('Importing {} from {}'.format(file, location))

    df = pd.read_csv(path)
    df = df.sample(n=500) # Remove if the whole df is to be used

    df['audio'] = [speech_file_to_array(os.path.join(location, x)) for x in df['wav_filename']]
    df = df.drop(columns=['wav_filename'])
    df = df.drop(columns=['wav_filesize'])

    return df