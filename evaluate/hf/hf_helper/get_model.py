import torch

from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModelForPreTraining, HubertForCTC

def get_model(model_name):
    print('GETTING MODEL: {}'.format(model_name))
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model = Wav2Vec2ForCTC.from_pretrained(model_name).to(device)
    processor = Wav2Vec2Processor.from_pretrained(model_name)

    print('LOADED MODEL')
    print('USING DEVICE: {}'.format(device))

    return model, processor, device

    