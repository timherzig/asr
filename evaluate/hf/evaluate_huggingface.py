import gc
import torch

from ..metrics.wer import wer
from ..metrics.cer import cer
from datasets import Dataset
from .hf_helper.get_model import get_model

def evaluate_huggingface(m, ds):
    print('EVALUATING: huggingface')

    model, processor, device = get_model(m)
    dataset = Dataset.from_pandas(ds)

    del ds

    gc.collect()

    def evaluate(batch):
            inputs = processor(batch['audio'], sampling_rate=16_000, return_tensors="pt", padding=True)

            with torch.no_grad():
                if 'base' in m:
                    logits = model(inputs.input_values.to(device)).logits
                else:
                    logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits

            pred_ids = torch.argmax(logits, dim=-1)
            batch['prediction'] = processor.batch_decode(pred_ids)
            return batch
    
    print('STARTED EVALUATING ...')

    result = dataset.map(evaluate, batched=True, batch_size=8)

    predictions = [x.upper() for x in result['prediction']]
    references = [x.upper() for x in result['transcript']]

    m_wer = wer(predictions=predictions, references=references, chunk_size=500) * 100

    del model, processor, device, dataset

    gc.collect()

    return m_wer