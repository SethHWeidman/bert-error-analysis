import copy
import collections
import os
import random
import typing

import torch
from torch.utils import data
from torchtext import vocab
import transformers

import preprocess

DATA_DIR = os.getcwd()
TOKENIZER = transformers.AutoTokenizer.from_pretrained('bert-base-uncased', unk_token='<unk>')

Sentence = typing.List[str]
Paragraph = typing.List[Sentence]


class BERTWiki2Dataset(data.Dataset):
    def __init__(self, data_dir, max_vocab_size, max_seq_len):
        paragraphs = _read_wikitext_2(data_dir)
        self.max_vocab_size = max_vocab_size
        self.max_seq_len = max_seq_len
        self.data, self.vocab = self._paragraphs_to_vocab_and_data(paragraphs)

        examples = []
        for paragraph in self.data:
            examples.extend(self._get_next_sentence_data_from_paragraph(paragraph))
        examples = [
            (self._get_masked_language_model_data_from_tokens(tokens) + (segments, is_next))
            for tokens, segments, is_next in examples
        ]
        # Pad inputs
        (
            self.all_token_ids,
            self.all_segments,
            self.valid_lens,
            self.all_pred_positions,
            self.all_mlm_weights,
            self.all_mlm_labels,
            self.nsp_labels,
        ) = self._pad_bert_inputs(examples)

    def __getitem__(self, idx) -> typing.Tuple:
        return (
            self.all_token_ids[idx],
            self.all_segments[idx],
            self.valid_lens[idx],
            self.all_pred_positions[idx],
            self.all_mlm_weights[idx],
            self.all_mlm_labels[idx],
            self.nsp_labels[idx],
        )

    def __len__(self):
        return len(self.all_token_ids)

    def _paragraphs_to_vocab_and_data(self, paragraphs: typing.List[Paragraph]) -> vocab.Vocab:
        # tokenize paragraphs
        paragraphs_tokenized = [self._tokenize_paragraph(paragraph) for paragraph in paragraphs]
        counter = collections.Counter()
        for tokenized_paragraph in paragraphs_tokenized:
            for sentence in tokenized_paragraph:
                for token in sentence:
                    counter[token] += 1
        return (
            paragraphs_tokenized,
            vocab.Vocab(
                counter,
                max_size=self.max_vocab_size,
                min_freq=5,
                specials=['<pad>', '<mask>', '<cls>', '<sep>'],
            ),
        )

    def _tokenize_paragraph(self, paragraph: Paragraph) -> Paragraph:
        return [TOKENIZER.tokenize(sentence) for sentence in paragraph]

    def _get_next_sentence_data_from_paragraph(self, paragraph: Paragraph) -> typing.List:
        next_sentence_data_from_paragraph = []
        for i in range(len(paragraph) - 1):
            tokens_a, tokens_b, is_next = self._get_next_sentence(paragraph[i], paragraph[i + 1])

            # simply exclude sentences that are too long, accounting for special tokens
            if len(tokens_a) + len(tokens_b) + 3 > self.max_seq_len:
                continue
            tokens, segments = preprocess.get_tokens_and_segments(tokens_a, tokens_b)
            next_sentence_data_from_paragraph.append((tokens, segments, is_next))
        return next_sentence_data_from_paragraph

    def _get_next_sentence(self, sentence: Sentence, next_sentence: Sentence) -> typing.Tuple:
        '''
        Helper function to return either the next sentence (passed in) or a random sentenc
        From https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html 
        '''
        r = random.random()
        if r < 0.5:
            is_next = True
        else:
            # randomly select a "next sentence"
            next_sentence = random.choice(random.choice(self.data))
            is_next = False
        return sentence, next_sentence, is_next

    def _get_masked_language_model_data_from_tokens(
        self, tokens: typing.List[str]
    ) -> typing.Tuple:
        candidate_pred_positions = []
        for i, token in enumerate(tokens):
            if token in ['<cls>', '<sep>']:
                continue
            candidate_pred_positions.append(i)
        # select 15% of tokens to predict
        num_mlm_preds = max(1, round(len(tokens) * 0.15))
        mlm_input_tokens, pred_positions_and_labels = self._replace_mask_language_model_tokens(
            tokens, candidate_pred_positions, num_mlm_preds
        )
        # sort by
        pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        pred_positions = [v[0] for v in pred_positions_and_labels]
        mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        return (
            [self.vocab[token] for token in mlm_input_tokens],
            pred_positions,
            [self.vocab[token] for token in mlm_pred_labels],
        )

    def _replace_mask_language_model_tokens(
        self,
        tokens: typing.List[str],
        candidate_pred_positions: typing.List[int],
        num_mlm_preds: int,
    ) -> typing.Tuple:
        '''
        Returns a list of tokens, some of which have now been masked / replaced, along with a list
        of the indices of the positions that we are going to predict, and the original tokens 
        associated with those indices
        '''
        # make a copy to not overwrite original
        mlm_tokens = copy.deepcopy(tokens)
        pred_positions_and_labels = []

        # shuffle to ensure random selection of tokens
        random.shuffle(candidate_pred_positions)
        for mlm_pred_position in candidate_pred_positions:
            if len(pred_positions_and_labels) >= num_mlm_preds:
                break
            r = random.random()
            if r < 0.8:
                masked_token = '<mask>'
            elif r >= 0.8 and r < 0.9:
                masked_token = tokens[mlm_pred_position]
            else:
                masked_token = self._pick_random_token()

            mlm_tokens[mlm_pred_position] = masked_token
            pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))

        return mlm_tokens, pred_positions_and_labels

    def _pick_random_token(self) -> str:
        # exclude special characters
        return self.vocab.itos[random.randint(4, len(self.vocab) - 1)]

    def _pad_bert_inputs(self, examples: typing.List[typing.Tuple]) -> typing.Tuple:
        max_num_mlm_preds = round(self.max_seq_len * 0.15)
        all_token_ids, all_segments, valid_lens = [], [], []
        all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
        nsp_labels = []
        for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
            num_token_ids = len(token_ids)
            num_mlm_pred_label_ids = len(mlm_pred_label_ids)
            rem_pred_positions = max_num_mlm_preds - len(pred_positions)
            all_token_ids.append(
                torch.tensor(
                    token_ids + [self.vocab['<pad>']] * (self.max_seq_len - num_token_ids)
                )
            )
            all_segments.append(torch.tensor(segments + [0] * (self.max_seq_len - len(segments))))
            # `valid_lens` excludes counts of '<pad>' tokens
            valid_lens.append(torch.tensor(num_token_ids))
            all_pred_positions.append(torch.tensor(pred_positions + [0] * (rem_pred_positions)))
            # assign 0 weight to the padded elements
            all_mlm_weights.append(
                torch.cat([torch.ones(num_mlm_pred_label_ids), torch.zeros(rem_pred_positions)])
            )
            all_mlm_labels.append(
                torch.tensor(
                    mlm_pred_label_ids + [0] * (max_num_mlm_preds - num_mlm_pred_label_ids)
                )
            )
            nsp_labels.append(torch.tensor(is_next))
        return (
            all_token_ids,
            all_segments,
            valid_lens,
            all_pred_positions,
            all_mlm_weights,
            all_mlm_labels,
            nsp_labels,
        )


def _read_wikitext_2(data_dir: str) -> typing.List[Paragraph]:
    '''
    Reads in the wikitext-2 dataset into a list of "Paragraph"s: lists of lists of strings
    '''
    filename = os.path.join(data_dir, 'wiki.train.tokens')
    with open(filename, 'r') as f:
        lines = f.readlines()

    # return the file as a list of lists of sentences, split on ' . ' because that the punctuation
    # marks are split out from the words in the file itself
    paragraphs = [
        line.strip().lower().split(' . ') for line in lines if len(line.split(' . ')) >= 2
    ]
    random.shuffle(paragraphs)
    return paragraphs
