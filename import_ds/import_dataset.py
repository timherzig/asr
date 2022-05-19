import pandas as pd
from .import_cv.import_commonvoice import import_commonvoice
from .import_mls.import_mls import import_mls

# Returns a pandas dataset containing columns 
#     - audio
#     - transcript

def clean_transcript(x):
    x = x.lower()
    x = x.replace('ü', 'ue')
    x = x.replace('ä', 'ae')
    x = x.replace('ö', 'oe')
    x = x.replace('ß', 'ss')
    return x

def import_dataset(path, remove_special_chars):
    print('IMPORTING...')
    
    if ('cv' or 'CV') in path:
        df = import_commonvoice(path)
    if ('mls' or 'MLS') in path:
        df = import_mls(path)

    if remove_special_chars:
        print('cleaning dataset')
        df['target_sentence'] = [clean_transcript(x) for x in df['transcript']]
        df.drop(columns=['transcript'], inplace=True)
        df.rename(columns={'target_sentence': 'transcript'}, inplace=True)

    return df