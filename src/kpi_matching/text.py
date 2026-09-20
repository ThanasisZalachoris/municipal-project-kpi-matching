"""Versioned Unicode tokenization with an explicit small English stopword list."""

import re
import unicodedata

STOPWORDS = frozenset('a an the and or of to for in with on by as at is are be from'.split())
PREPROCESSING_VERSION = 'unicode-word-v1'


def tokens(text):
    if not isinstance(text, str):
        raise ValueError('Text must be a string')
    if len(text) > 10000:
        raise ValueError('Text exceeds supported limit')
    normalized = unicodedata.normalize('NFKC', text).casefold()
    return [w for w in re.findall(r'[^\W_]+', normalized) if len(w) >= 2 and w not in STOPWORDS]
