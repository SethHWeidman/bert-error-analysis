import typing

import torch
from torch.utils import data as utils_data
from torchtext import vocab
import transformers

import preprocess


sentences = '''The Rock is destined to be the 21st Century 's new `` Conan '' and that he 's going to make a splash even greater than Arnold Schwarzenegger , Jean-Claud Van Damme or Steven Segal .
The gorgeously elaborate continuation of `` The Lord of the Rings '' trilogy is so huge that a column of words can not adequately describe co-writer\/director Peter Jackson 's expanded vision of J.R.R. Tolkien 's Middle-earth .'''
sentences_list = sentences.split('\n')
labels = [0, 1]

class SentimentAnalysisDataset(utils_data.Dataset):
    def __init__(
        self, voc: vocab.Vocab, tokenizer: transformers.PreTrainedTokenizer, max_seq_len: int
    ) -> None:
        self.sentences = sentences_list  ## TODO: make this read in the actual data
        self.labels = [0, 1]
        self.vocab = voc
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        examples = []
        weights = []
        segments = []
        labels = []

        for sentence, label in zip(self.sentences, self.labels):
            sentence_tokens, segment_tokens = self._get_sentence_tokens(sentence)
            sentence_tokens_tensor, sentence_weights_tensor = self.get_sentence_token_tensors(
                sentence_tokens
            )
            examples.append(sentence_tokens_tensor)
            weights.append(sentence_weights_tensor)
            segments.append(segment_tokens)
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

    def get_sentence_token_tensors(self, tokens: str) -> torch.Tensor:
        sentence_length = len(tokens)
        return (
            torch.tensor(tokens + [0] * (self.max_seq_len - sentence_length)),
            torch.cat(
                [torch.ones(sentence_length), torch.zeros(self.max_seq_len - sentence_length)]
            ),
        )

    def _get_sentence_tokens(self, sentence: str) -> typing.List[int]:
        sentence_tokens = self.tokenizer.tokenize(sentence)
        sentence_tokens_processed, segment_tokens = preprocess.get_tokens_and_segments(
            sentence_tokens
        )
        return ([self._get_token(token) for token in sentence_tokens_processed], segment_tokens)

    def _get_token(self, s: str) -> str:
        try:
            return self.vocab.stoi[s]
        except KeyError:
            return self.vocab.stoi['<unk>']
