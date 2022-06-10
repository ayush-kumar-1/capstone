import pandas as pd 
import numpy as np

from tokenizers import Tokenizer, normalizers
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.models import WordPiece as WP
from tokenizers.normalizers import NFD, StripAccents, Lowercase
from tokenizers.trainers import WordPieceTrainer

import emoji
import preprocessor as pre


def load_data(path): 
    """
    Loads the data from the given path. 
    """
    return pd.read_csv(path, index_col=0)

def label_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converts the labels to numeric values. 
    """
    cat_dict = {}
    for i, type in enumerate(df.label.unique()): 
        cat_dict[type] = i

    return df.label.map(cat_dict)

def preprocess(tweet_arr: pd.Series) -> pd.Series:
    """
    Preprocesses the text. 
    """
    pre.set_options(pre.OPT.URL, pre.OPT.MENTION, pre.OPT.RESERVED, pre.OPT.SMILEY)

    tweet_arr = tweet_arr.apply(emoji.demojize)
    tweet_arr = tweet_arr.apply(pre.tokenize)
    
    return tweet_arr


def tokenize(tweet_arr: pd.Series) -> pd.Series:
    """
    Tokenizes the text. 
    """
    tokenizer = Tokenizer(WP())
    normalizer = normalizers.Sequence([Lowercase(), NFD(), StripAccents()])

    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.normalizer = normalizer

    special_tokens = ["$HASHTAG$", "$EMOJI$", "$URL$", "$RESERVED$", "$MENTION$", "[UNK]"]
    trainer = WordPieceTrainer(special_tokens = special_tokens)

    files = tweet_arr.to_list()
    tokenizer.train_from_iterator(files, trainer)

    tokens = pd.Series([encoding.tokens for encoding in tokenizer.encode_batch(tweet_arr)])

    return tokens