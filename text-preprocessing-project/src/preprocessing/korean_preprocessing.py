import os
import re
import json
import pandas as pd
import unicodedata
from konlpy.tag import Okt
from pykospacing import Spacing
from soynlp.tokenizer import LTokenizer

okt = Okt()
spacing = Spacing()
soynlp_tokenizer = LTokenizer()

korean_stopwords = set([
    '이', '그', '저', '것', '수', '등', '들', '및', '에서', '에게', '으로', '하고', '보다', '도', 
    '는', '은', '가', '을', '를', '의', '에', '와', '과', '이다', '있다', '되다', '합니다', '그리고'
])

def clean_korean_text(text):
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"[^가-힣a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def remove_korean_stopwords(tokens):
    return [word for word in tokens if word not in korean_stopwords and len(word) > 1]

def preprocess_sentence_korean(sentence):
    cleaned = clean_korean_text(sentence)
    tokens_with_pos = okt.pos(cleaned, norm=True, stem=True)
    tokens = [word for word, pos in tokens_with_pos if pos in ['Noun', 'Verb', 'Adjective']]
    tokens = remove_korean_stopwords(tokens)
    return " ".join(tokens)

def correct_spacing(text):
    return spacing(text)

def tokenize_with_soynlp(text):
    return soynlp_tokenizer.tokenize(text)

def preprocess_korean_text(text):
    text_with_corrected_spacing = correct_spacing(text)
    preprocessed_text = preprocess_sentence_korean(text_with_corrected_spacing)
    return preprocessed_text

def preprocess_all_korean_texts(texts):
    return [preprocess_korean_text(text) for text in texts]