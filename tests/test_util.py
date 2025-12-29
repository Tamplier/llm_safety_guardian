from unittest.mock import patch
import pytest
import spacy
import numpy as np
from sklearn.metrics import f1_score
from sentence_transformers import SentenceTransformer
from src.transformers import SpacyTokenizer, SbertVectorizer
from src.util import (
  CachingSpellChecker, typos_processor, PathHelper,
  find_threshold, filter_by_threshold
)

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
    assert 'llm_safety_guardian' in str(PathHelper.project_root.resolve())
    assert 'models/' in str(PathHelper.models.label_encoder.resolve())
    assert 'data/processed/' in str(PathHelper.data.processed.x_train.resolve())

@pytest.mark.parametrize(
    'y,prob,te,ce,mask_expected',
    [
        ([1,0,1,0],
         [[0.05, 0.95],
          [0.95, 0.05],
          [0.4, 0.6],
          [0.45, 0.55]],
          0.6,
          0.75,
          [True, True, True, False]),
        ([1,0,1,0],
         [[0.05, 0.95],
          [0.95, 0.05],
          [0.4, 0.6],
          [0.55, 0.45]],
          0.5,
          1,
          [True, True, True, True])
    ]
)
def test_confidence_threshold(y, prob, te, ce, mask_expected):
    y = np.array(y)
    prob = np.array(prob)
    t, c = find_threshold(y, prob, f1_score)
    assert t == te
    assert c == ce
    mask, _ = filter_by_threshold(prob, t)
    np.testing.assert_array_equal(mask, mask_expected)

def test_pickle_base(monkeypatch):
    counting_init(SentenceTransformer, monkeypatch)
    SbertVectorizer._bo_storage.clear()
    tokenizer = SpacyTokenizer()
    vectorizer = SbertVectorizer()
    assert tokenizer.nlp is not None
    assert vectorizer.model is not None
    assert vectorizer.tokenizer is not None
    assert SentenceTransformer.created == 1

    vectorizer2 = SbertVectorizer()
    vectorizer2.model
    assert SentenceTransformer.created == 1

    t_objects = list(tokenizer._bo_storage.keys())
    v_objects = list(vectorizer._bo_storage.keys())
    np.testing.assert_array_equal(t_objects, ['nlp'])
    np.testing.assert_array_equal(v_objects, ['model', 'tokenizer'])

def counting_init(c, monkeypatch):
    c.created = 0
    original_init = c.__init__
    def custom_init(self, *args, **kwargs):
        type(self).created += 1
        original_init(self, *args, **kwargs)
    monkeypatch.setattr(c, "__init__", custom_init)
