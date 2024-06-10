# AAdaM at SemEval-2024 Task 1: Augmentation and Adaptation for Multilingual Semantic Textual Relatedness

This repository contains code for our system paper: 
[AAdaM at SemEval-2024 Task 1: Augmentation and Adaptation for Multilingual Semantic Textual Relatedness](https://arxiv.org/abs/2404.01490). If you find this repository useful, please consider citing our paper.

Contact: Miaoran Zhang (mzhang@lsv.uni-saarland.de)

## Setup
Create your docker image or virtual environment based on the provided docker file ([Dockerfile](https://github.com/uds-lsv/AAdaM/blob/master/Dockerfile)). Specifically,
```
python: 3.8
torch: 1.14.0
CUDA: 11.8.0
```

## Code

### Training
- `train_mlm.py`: MLM with full fine-tuning.
- `train_ft.py`: Supervised learning with full fine-tuning (supports warmup).
- `train_lang_adapter.py`: Train language adapters (supports continued pre-training).
- `train_task_adapter.py`: Train task adapters.
- `train_warmup_adapter`: Train task adapters on noisy data first, then on provided task data.

### Evaluation
- `eval_ft.py`: Evaluate full fine-tuned models.
- `eval_adapters.py`: Evaluate adapters.
- `eval_fasttext.py`: Evaluate FastText embeddings.
- `eval_overlap.py`: Evaluate lexical overlap.
- `eval_sentemb.py`: Evaluate out-of-the-box embeddings extracted from pre-trained models.

### Others
- `prepare_data.py`: Prepare data for various scenarios, e.g., 
  - Split task data to n-folds for cross-validation.
  - Extract unlabeled sentences for task-adaptive pre-training.
  - Prepare downloaded Leizig data for training language adapters.
- `utils.py`: Utilities functions, e.g., data processing, computing language similarites.

Please check [scripts](https://github.com/uds-lsv/AAdaM/tree/master/scripts) for experimentation details. 

