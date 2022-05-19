import sys, gc
import argparse
from os import remove

from import_ds.import_dataset import import_dataset
from evaluate.evaluate_main import evaluate_main
from train.train_main import train_main

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('-task', type=str, help='Task to perform: evaluate, train')
    parser.add_argument('-toolkit', type=str, help='Toolkit to use: deepspeech, huggingface, fairseq')

    parser.add_argument('-dataset', type=str, help='Dataset to use (path to root folder)')
    parser.add_argument('-model', type=str, help='Model to use (path to model location)')
    parser.add_argument('-remove_special_chars', type=bool, default=False)
    parser.add_argument('-n_runs', type=int, default=1)
    return parser.parse_args()


def main():
    print('IN MAIN APPLICATION')

    args = parse_arguments()
    dataset = args.dataset
    model_name = args.model
    task = args.task
    toolkit = args.toolkit
    remove_special_chars = args.remove_special_chars
    n_runs = args.n_runs
    wer = 0

    for i in range(0, n_runs):
        ds = import_dataset(dataset, remove_special_chars) #pandas dataset containing columns ['audio', 'transcript']

        if args.task == 'evaluate':
            wer += evaluate_main(toolkit, ds, model_name)
        if args.task == 'train':
            train_main(toolkit, ds, model_name)

        del ds
        gc.collect()

    print('Model: {}'.format(model_name))
    print('Dataset: {}'.format(dataset))
    print('AVG WER: {} for {} runs'.format(wer/n_runs, n_runs))
if __name__ == "__main__":
    main()