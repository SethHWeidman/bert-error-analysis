from typing import List, Optional, Tuple


def get_tokens_and_segments(tokens_a: List[str], tokens_b: Optional[List[str]] = None) -> Tuple:
    '''
    Takes in the tokens from either one sentences or two and returns the "BERT representation"
    of the sentence, with the '<cls>' and '<sep>' tokens in the right places.
    Taken from https://d2l.ai/chapter_natural-language-processing-pretraining/bert.html
    '''
    tokens = ['<cls>'] + tokens_a + ['<sep>']
    # 0 for segment 1, 1 for segment 2
    segments = [0] * (len(tokens_a) + 2)
    if tokens_b is not None:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b) + 1)
    return tokens, segments
