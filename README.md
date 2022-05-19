# asr

Program to perform ASR tasks for research

Tasks supported:

  - Evaluation

Toolkits supported:

  - Hugging Face
  - DeepSpeech

Extensions being worked on:

  - Training


Usage guide: 

Run main.py in the root folder with the following arguments:

- '-task': task to be performed (evaluate, train)
- '-toolkit': toolkit to be used (huggingface, deepspeech)
- '-dataset': dataset to be used, see further instructions in python-files
- '-model': model to be used
- '-remove_special_chars': True is dataset need cleaning/removal of unwanted chars
- '-n_runs': if task is to be performed multiple times
