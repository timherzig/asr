from .hf.evaluate_huggingface import evaluate_huggingface


def evaluate_main(toolkit, dataset, model_name):
    print('TASK BEING PERFORMED: evaluate')
    wer = 0

    if toolkit == 'deepspeech':
        print('TOOLKIT: deepspeech')
    
    if toolkit == 'huggingface':
        print('TOOLKIT: huggingface')
        wer = evaluate_huggingface(model_name, dataset)

    return wer