# bert-error-analysis

This repon contains code to produce a CSV where each row is an observation from the Stanford Sentiment Treebank dataset, along with the predictions from an LSTM, and the predictions from a RoBERTa model fine-tuned on this dataset using the [`transformers`](https://huggingface.co/transformers/) library.

I presented the results from this in my ODSC West 2020 talk on swap-ins / swap-outs, since this code allows you to not only produce the accuracy numbers from each dataset, but also to see which individual observations were predicted correctly by one model but incorrectly by another model.

# Setup

## Installation

This assumes you have `conda` installed.

1. `conda update python`
2. `conda create -n bert-error-analysis python=3.8.1`
3. `pip install -r requirements.txt`

### Other

* Change `BASE_DIR` in `const.py` to be the folder of this repo on your computer.

## Fine-tuning RoBERTa on the Stanford Sentiment Treebank dataset

The [Stanford Sentiment Treebank dataset](https://nlp.stanford.edu/sentiment/treebank.html) 
contains, among other things, a labeled dataset of ~11K sentences, each with an associated 
sentiment score between 0 and 1 and a "split" determining whether the sentence is in the training
set or the evaluation set. 

`cd`ing into `main` and running `python fine_tune_pretrained_three_class.py` will:

1. Use the [`transformers` library](https://huggingface.co/transformers/) to load in the `roberta-base` tokenizer and the data
itself into a PyTorch  [Dataset](https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset). Using the settings  in this Python file, the raw labels in the dataset - which are continuous sentiment scores from 0 to 1 - will be mapped to three labels:

* `0` (negative): sentiment score less than 0.4
* `1` (neutral): sentiment score greater than 0.4 (inclusive) and less than 0.6 (exclusive)
* `2` (positive): sentiment score greater than 0.6

2. Initialize a "`RobertaFineTuningModel`", which contains the `roberta-base` model from  `transformers` and a fully connected layer on top of the final representation for the `[CLS]` token (which itself comes from the `pooler` of the [`RobertaModel`](https://huggingface.co/transformers/model_doc/roberta.html#robertamodel) - for more on this, see [here](https://github.com/google-research/bert/issues/43)).

3. Train this for three epochs on the sentiment analysis data, saving the resulting models at each epoch. By default, the script runs this using the smaller of the two batch sizes mentioned in the  RoBERTa paper, `16`, and the middle learning rate of the three mentioned, `2e-5`. By default, it fine-tunes for five epochs (following the paper) and saves the model results each epoch.
Warning: *each* of these models will take up between 400 and 450 MB once saved!

4. Then, once trained, you can run `python evaluation.py`, pointing to the correct model folder in that script, to generate a CSV containing the predictions for each observation. 

## Pre-training BERT

To pre-train a toy version of BERT with just under 5M parameters for 50 epochs on the Wiki-2  dataset, `cd` into `main`, and run:

```
python pretrain_custom.py
```

This will save lists of the losses for the masked language modeling and next sentence prediction tasks. You can plot these losses over time by running:

```
python plot.py
```

## References

[RoBERTa paper](https://arxiv.org/abs/1907.11692)

[Helpful guide for pre-training BERT](https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html) (uses MXNet)