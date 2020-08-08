from os import path
import typing

import torch
from torch.utils import data as utils_data
from torchtext import vocab
import transformers

import const
import preprocess


BASE_DATA_PATH = path.join(const.BASE_DIR, 'data', 'stanfordSentimentTreebank')


class SentimentAnalysisDataset(utils_data.Dataset):
    def __init__(
        self, voc: vocab.Vocab, tokenizer: transformers.PreTrainedTokenizer, max_seq_len: int
    ) -> None:
        sentences, labels, splits = read_stanford_sentiment(BASE_DATA_PATH)
        self.sentences = sentences
        self.labels = [1 if label >= 0.5 else 0 for label in labels]
        self.vocab = voc
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        examples = []
        weights = []
        segments = []
        labels = []

        for sentence, label in zip(self.sentences, self.labels):
            sentence_tokens, segment_tokens = self._get_sentence_tokens(sentence)
            (
                sentence_tokens_tensor,
                sentence_weights_tensor,
                segments_tensor,
            ) = self.get_sentence_token_tensors(sentence_tokens, segment_tokens)
            examples.append(sentence_tokens_tensor)
            weights.append(sentence_weights_tensor)
            segments.append(segments_tensor)
            labels.append(torch.tensor(label, dtype=torch.int64))

        self.examples = examples
        self.weights = weights
        self.segments = segments
        self.labels = labels

    def __getitem__(self, idx: int) -> typing.Tuple:
        return (
            self.examples[idx],
            self.weights[idx],
            self.segments[idx],
            self.labels[idx],
        )

    def __len__(self) -> int:
        return len(self.examples)

    def get_sentence_token_tensors(
        self, sentence_tokens: typing.List[int], segment_tokens: typing.List[int]
    ) -> torch.Tensor:
        sentence_length = len(sentence_tokens)
        return (
            torch.tensor(tokens + [0] * (self.max_seq_len - sentence_length)),
            torch.cat(
                [torch.ones(sentence_length), torch.zeros(self.max_seq_len - sentence_length)]
            ),
            torch.tensor(segment_tokens + [0] * (self.max_seq_len - len(segment_tokens))),
        )

    def _get_sentence_tokens(self, sentence: str) -> typing.List[int]:
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_tokens_processed, segment_tokens = preprocess.get_tokens_and_segments(
            sentence_tokens
        )
        return [self._get_token(token) for token in sentence_tokens_processed], segment_tokens

    def _get_token(self, s: str) -> str:
        try:
            return self.vocab.stoi[s]
        except KeyError:
            return self.vocab.stoi['<unk>']


def read_stanford_sentiment(base_data_path: str) -> typing.Tuple[typing.List]:
    '''
    Reads in the stanford sentiment dataset into a tuple of three lists:
    * The sentences, as strings
    * The sentiment labels, as floats between 0 and 1
    * The dataset "split": train vs. validation
    '''
    with open(path.join(BASE_DATA_PATH, 'datasetSentences.txt'), 'r') as f:
        lines = f.read().splitlines()
    sentence_partial_index_dict = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        index, sentence = line.split('\t')
        sentence_partial_index_dict[sentence] = index
    sentences_list = list(sentence_partial_index_dict.keys())

    with open(path.join(BASE_DATA_PATH, 'dictionary.txt'), 'r') as f:
        lines = f.read().splitlines()
    dictionary_strings = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        string, index = line.split('|')
        dictionary_strings[string] = index

    sentence_overall_index_dict = {}
    for sentence in sentences_list:
        try:
            sentence_overall_index_dict[sentence] = dictionary_strings[sentence]
        except:
            pass

    with open(path.join(BASE_DATA_PATH, 'sentiment_labels.txt'), 'r') as f:
        lines = f.read().splitlines()
    sentence_part_labels = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        index, label = line.split('|')
        sentence_part_labels[index] = label

    with open(path.join(BASE_DATA_PATH, 'datasetSplit.txt'), 'r') as f:
        lines = f.read().splitlines()
    dataset_split_dict = {}
    for i, line in enumerate(lines):
        if i == 0:
            continue
        index, split = line.split(',')
        dataset_split_dict[index] = split

    sentences_final = []
    labels_final = []
    splits_final = []
    for sentence in sentences_list:
        try:
            partial_index = sentence_partial_index_dict[sentence]
            overall_index = sentence_overall_index_dict[sentence]
        except KeyError:
            continue
        sentences_final.append(sentence)
        split = dataset_split_dict[partial_index]
        label = sentence_part_labels[overall_index]
        labels_final.append(float(label))
        splits_final.append(int(split))

    print(len(sentences_final))
    print(len(labels_final))
    print(len(splits_final))
    assert len(sentences_final) == len(labels_final) == len(splits_final) == 11272
    return sentences_final, labels_final, splits_final
