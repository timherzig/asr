

def evaluate_main(toolkit, dataset, model_name, scorer_name):
    print('TASK BEING PERFORMED: evaluate')
    wer = 0

    if toolkit == 'deepspeech':
        from .ds.evaluate_deepspeech import evaluate_deepspeech

        print('TOOLKIT: deepspeech')
        wer = evaluate_deepspeech(model_name, scorer_name, dataset)
    
    if toolkit == 'huggingface':
        from .hf.evaluate_huggingface import evaluate_huggingface
        
        print('TOOLKIT: huggingface')
        wer = evaluate_huggingface(model_name, dataset)

    return wer