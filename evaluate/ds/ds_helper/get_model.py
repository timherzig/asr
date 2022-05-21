import os
import deepspeech

def get_model(model_name, scorer_name):
    print('GETTING MODEL: {}'.format(model_name))
    
    if(os.path.exists(model_name)):
        model = deepspeech.Model(model_name)
    else:
        print('Model {} not found, exiting...'.format(model_name))
    
    if (scorer_name != '-1') and (os.path.exists(scorer_name)):
        model.enableExternalScorer(scorer_name)
    else:
        print('Scorer not found, continuing without one')

    print('LOADED MODEL')

    return model
