import math
from sklearn.preprocessing import FunctionTransformer
import pytest
import numpy as np
import pandas as pd
from src.transformers import (
    fix_concatenated_words,
    SpacyTokenizer, FeatureSelector,
    SbertVectorizer
)

@pytest.mark.parametrize(
    'text,expected',
    [
        ('HELP!HELP! I need some help!', 'HELP! HELP! I need some help!'),
        ('TEST//TEST//TEST', 'TEST// TEST// TEST'),
        ('BEEP||||BEEP|||BEEP||BEEP', 'BEEP|||| BEEP||| BEEP|| BEEP'),
        ("Don't do that", "Don't do that")
    ]
)
def test_sentence_splitter(text, expected):
    splitter = FunctionTransformer(fix_concatenated_words, validate=False)
    transformed = splitter.transform([text])[0]
    assert transformed == expected

@pytest.mark.parametrize(
    'text,expected',
    [
        ('HI!!! My name is Jonas!! What is your name?????', 3),
        ("Don't do that/", 1),
        ('Something, something, something \n\n Darkside!! \n Something, something, something \n \n Completely', 4)
    ]
)
def test_spacy_tokenizer(text, expected):
    tokenizer = SpacyTokenizer()
    doc = tokenizer.transform([text])[0]
    # According to spacy it's ok to have empty sentece
    sentences_len = np.array([len(sent.text.strip()) for sent in doc.sents])
    sentences_len = sentences_len[sentences_len > 0]
    count = len(sentences_len)
    assert count == expected

def test_feature_selector():
    np.random.seed(42)
    sample_size = 50
    x = np.random.uniform(low=0, high=1, size=sample_size)
    y = np.sin(x)
    a = y + np.random.normal(loc=0, scale=2, size=sample_size)
    b = y - np.random.normal(loc=50, scale=5, size=sample_size)
    c = np.random.uniform(low=-100, high=100, size=sample_size)
    d = np.zeros(sample_size)
    X = pd.DataFrame({'a': a, 'b': b, 'c': c, 'd': d})
    selector = FeatureSelector(top_k=2)
    result = selector.fit_transform(X, y)
    np.testing.assert_array_equal(np.sort(list(result.columns)), ['a', 'b'])

vectorizer = SbertVectorizer()

@pytest.mark.parametrize(
    'text,chunks',
    [
        ('Small text. Chunks != sentences. But to a first approximation, they can be.', 1),
        ('Very loooong text. '*150, 150),
        ('STOP!!! '*500, 500)
    ]
)
def test_sbert_vectorizer_chunk_text_by_tokens(text, chunks):
    all_mpnet_base_max_input = vectorizer.model.max_seq_length
    max_chunk_size = vectorizer.chunk_token_size
    token_len = vectorizer._token_length(text)
    result = vectorizer._chunk_text_by_tokens(text)
    assert all_mpnet_base_max_input > max_chunk_size > 1
    assert token_len > 0
    assert len(result) == chunks

def test_sbert_vecorizer_transform():
    text = 'Small text to test vectorizer'
    result = vectorizer.transform([text])[0]
    assert len(result) == 768
