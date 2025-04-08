import pandas as pd
from soynlp.tokenizer import MaxTokenizer
from soynlp.normalizer import normalize

def tokenize(text):
    """
    Tokenizes the input text using SOYNLP's MaxTokenizer.
    """
    tokenizer = MaxTokenizer()
    tokens = tokenizer.tokenize(text)
    return tokens

def normalize_text(text):
    """
    Normalizes the input text using SOYNLP's normalization function.
    """
    normalized_text = normalize(text)
    return normalized_text

def preprocess_soynlp(text):
    """
    Preprocesses the input text using SOYNLP utilities.
    This includes tokenization and normalization.
    """
    normalized_text = normalize_text(text)
    tokens = tokenize(normalized_text)
    return tokens