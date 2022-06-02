from .ds.train_ds import train_ds
from .fairseq.train_fairseq import train_fairseq
from .hf.train_hf import train_hf
from .torch.train_torch import train_torch

def train_main(toolkit, dataset, model_name):
    print('TASK BEING PERFORMED: training')

    if toolkit == 'huggingface':
        print('{}'.format(toolkit))
        train_hf()

    elif toolkit == 'deepspeech':
        print('{}'.format(toolkit))
        train_ds()

    elif toolkit == 'fairseq':
        print('{}'.format(toolkit))
        train_fairseq()

    elif toolkit == 'torch':
        print('{}'.format(toolkit))
        train_torch()

