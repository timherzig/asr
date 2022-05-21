import os
import librosa
import warnings
import numpy as np
import pandas as pd

def speech_file_to_array_hf(root, x):
    file = os.path.join(root, 'audio', x.split('_')[0], x.split('_')[1], x + '.opus')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(file, sr=16_000, res_type='kaiser_fast')

    return speech_array

def speech_file_to_array_ds(root, x):
    file = os.path.join(root, 'audio', x.split('_')[0], x.split('_')[1], x + '.opus')
    
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        data, sample_rate = librosa.load(file, sr=16_000, res_type='kaiser_fast')
        int16 = (data * 32767).astype(np.int16)

    return int16

# Path should be the root folder (either test, train or dev) containing audio folder, segments.txt and transcript.txt
def import_mls(path, toolkit):
    print('Importing {} from {}'.format(path.split(os.sep)[-2], path.split(os.sep)[-3]))

    df = pd.read_csv(os.path.join(path, 'transcripts.txt'), delimiter='\t', header=None)
    df = df.sample(n=250) # remove for full dataset
    df.columns = ["path", "transcript"]

    if toolkit == 'huggingface':
        df['audio'] = [speech_file_to_array_hf(path, x) for x in df['path']]
    elif toolkit == 'deepspeech':
        df['audio'] = [speech_file_to_array_ds(path, x) for x in df['path']]
    
    return df
