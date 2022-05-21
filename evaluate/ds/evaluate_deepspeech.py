import gc
from webbrowser import get

from ..metrics.wer import wer
from ..metrics.cer import cer
from .ds_helper.get_model import get_model

def evaluate_deepspeech(m, s, ds):
    print('EVALUATING: deepspeech')

    model = get_model(m, s)

    ds['prediction'] = [model.stt(x) for x in ds['audio']]

    predictions = [x.upper() for x in ds['prediction']]
    transcripts = [x.upper() for x in ds['transcript']]

    m_wer = wer(predictions=predictions, references=transcripts, chunk_size=500) * 100

    del model, ds
    gc.collect()

    return m_wer