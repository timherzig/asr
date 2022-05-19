import os
import librosa
import warnings
import pandas as pd

def speech_file_to_array(root, x):
    file = os.path.join(root, 'audio', x.split('_')[0], x.split('_')[1], x + '.opus')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(file, sr=16_000, res_type='kaiser_fast')

    return speech_array

# Path should be the root folder (either test, train or dev) containing audio folder, segments.txt and transcript.txt
def import_mls(path):
    print('Importing {} from {}'.format(path.split(os.sep)[-2], path.split(os.sep)[-3]))

    df = pd.read_csv(os.path.join(path, 'transcripts.txt'), delimiter='\t', on_bad_lines='skip', header=None, nrows=10) # remove nrows arg. for full dataset
    df.columns = ["path", "transcript"]

    df['audio'] = [speech_file_to_array(path, x) for x in df['path']]

    return df
