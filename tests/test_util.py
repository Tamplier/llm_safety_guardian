from unittest.mock import patch
import pytest
import spacy
import numpy as np
from src.transformers import SpacyTokenizer, SbertVectorizer
from src.util import CachingSpellChecker, typos_processor, PathHelper

nlp = spacy.load('en_core_web_sm', disable=["ner", "textcat"])

@pytest.mark.parametrize(
    'words,expected',
    [
        (['Helpl', 'me', 'to', 'fimd', 'miself'], ['help', 'me', 'to', 'find', 'myself']),
    ]
)
def test_correct_words(words, expected):
    spell_checker = CachingSpellChecker()
    corrected = spell_checker.correct_words(words)
    np.testing.assert_array_equal(corrected, expected)

@pytest.mark.parametrize(
    'words,calls',
    [
        (['HECK', 'BEEP', 'BUMP', 'HECK', 'BUMP'], 3),
    ]
)
def test_correct_word(words, calls):
    with patch.object(CachingSpellChecker, '_correct_word', return_value=('a', 'b')) as mock_method:
        spell_checker = CachingSpellChecker()
        spell_checker.correct_words(words)
        assert mock_method.call_count == calls

@pytest.mark.parametrize(
    'text,expected',
    [
        ('FFFFFFFFF*CK!!!! It was soooooooo long t!me ago...', 'fuck!! it was so long time ago..'),
        ("Don't do that", "don't do that")
    ]
)
def test_typos_processor(text, expected):
    doc = nlp(text)
    transformed = typos_processor(doc)
    assert transformed == expected

def test_path_helper():
    assert 'nlp_suicide_watch' in str(PathHelper.project_root.resolve())
    assert 'models/' in str(PathHelper.models.label_encoder.resolve())
    assert 'data/processed/' in str(PathHelper.data.processed.x_train.resolve())


def test_pickle_base():
    tokenizer = SpacyTokenizer()
    vectorizer = SbertVectorizer()
    assert tokenizer.nlp is not None
    assert vectorizer.model is not None
    assert vectorizer.tokenizer is not None

    t_objects = list(tokenizer._bo_storage.keys())
    v_objects = list(vectorizer._bo_storage.keys())
    np.testing.assert_array_equal(t_objects, ['nlp'])
    np.testing.assert_array_equal(v_objects, ['model', 'tokenizer'])
