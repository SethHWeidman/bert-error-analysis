# bert-error-analysis

Comparing performance of BERT vs. other NLP models on individual cases from benchmark datasets

## Setup

### Installation

This assumes you have `conda` installed.

1. `conda update python`
2. `conda create -n bert-error-analysis python=3.8.1`
3. `pip install -r requirements.txt`

### Other

* Change `BASE_DIR` in `const.py` to be the folder of this repo on your computer.

## Overall organization

Scripts you'll want to run are in the `main` folder.

## Fine-tuning BERT on the Stanford Sentiment Treebank dataset

The [Stanford Sentiment Treebank dataset](https://nlp.stanford.edu/sentiment/treebank.html) 
contains, among other things, a labeled dataset of ~11K sentences, each with an associated 
sentiment score between 0 and 1 and a "split" determining whether the sentence is in the training
set or the evaluation set. 

`cd`ing into `main` and running `python fine_tune_pretrained_three_class.py` will:

1. Use the `transformers` library to load in the `bert-base-uncased` tokenizer and the data
itself into a PyTorch 
[Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Using the settings 
in this Python file, the raw labels in the dataset - which are continuous sentiment scores from 0
to 1 - will be mapped to three labels:

* `0` (negative): sentiment score less than 0.4
* `1` (neutral): sentiment score greater than 0.4 (inclusive) and less than 0.6 (exclusive)
* `2` (positive): sentiment score greater than 0.6

2. Initialize a "`BERTFineTuningModel`", which contains the `bert-base-uncased` model from 
`transformers` and a fully connected layer on top of the final representation for the `[CLS]` 
token (which itself comes from the `pooler` of the 
[`BertModel`](https://huggingface.co/transformers/model_doc/bert.html#bertmodel) - for more on 
this, see [here](https://github.com/google-research/bert/issues/43)).

3. Train this for three epochs on the sentiment analysis data, saving the resulting models at each 
epoch. By default, the script runs this for the four learning rates cited in the original BERT 
paper: `5e-5`, `4e-5`, `3e-5`, and `2e-5`. *Warning*: each of these models will take up between 400 
and 450 MB!

Then, once trained, you can run `python evaluation.py`, pointing to the correct model folder in 
that file, to generate a CSV containing the predictions for each observation. 

## Pre-training BERT

To pre-train a toy version of BERT with just under 5M parameters for 50 epochs on the Wiki-2 
dataset, `cd` into `main`, and run:

```
python pretrain_custom.py
```

This will save lists of the losses for the 

```
python plot.py
```

https://huggingface.co/transformers/model_doc/bert.html#bertmodel
