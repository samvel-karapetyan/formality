# Formality Evaluator

This is a codebase for evaluating model's formality. Understand how much they contains information about formality.

For evaluation is used `pavlick-formality-scores` dataset.

1. Pavlick, E., & Tetreault, J. (2016). *An Empirical Analysis of Formality in Online Communication*. *Transactions of the Association for Computational Linguistics*. Retrieved from https://www.aclweb.org/anthology/Q16-1010/
2. Lahiri, S. (2015). *SQUINKY! A Corpus of Sentence-level Formality, Informativeness, and Implicature*. arXiv preprint arXiv:1506.02306. Retrieved from https://arxiv.org/abs/1506.02306

## Setup

Create and activate the Conda environment:
   ```sh
   conda env create
   conda activate formality
   ```

## Running the Evaluation (Example)

1. Choose a model from HuggingFace.
2. Run the model using:
   ```sh
    python extractor.py --model_name=google-bert/bert-base-uncased --device=cuda:0 --batch_size=64 --num_workers=4
    python evaluator.py --model_name=google-bert/bert-base-uncased --n_repeats=5


| **Model Name**   |      **HuggingFace Path**            |    **Accuracy**       |
|------------------|--------------------------------------|-----------------------|
| **BERT U**       | `google-bert/bert-base-uncased`      |      87.69±0.39       |
| **BERT C**       | `google-bert/bert-base-cased`        |      89.29±0.31       |
| **ROBERTA**      | `FacebookAI/roberta-base`            |      88.96±0.36       |
| **DistilBERT**   | `distilbert/distilbert-base-uncased` |      89.19±0.22       |
| **ALBERT**       | `albert/albert-base-v2`              |      88.10±0.18       |
