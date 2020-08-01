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

## Pre-training BERT

To pre-train a toy version of BERT with just under 5M parameters for 50 epochs on the Wiki-2 
dataset, run:

```
python main.py
```

Then, to plot the loss over batches after running for 50 epochs, run:

```
python plot.py
```

