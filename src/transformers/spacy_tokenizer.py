import logging
import spacy
from sklearn.base import BaseEstimator, TransformerMixin
from spacy.language import Language
from emot.emo_unicode import EMOTICONS_EMO
from src.util.pickle_compatible import PickleCompatible
from src.util import GPUManager

logger = logging.getLogger(__name__)

class SpacyTokenizer(BaseEstimator, TransformerMixin, PickleCompatible, GPUManager):
    _big_objects = ['nlp']

    @staticmethod
    @Language.component("newline_sentencizer")
    def newline_sentencizer(doc):
        for token in doc:
            if '\n' in token.text and token.i > 0:
                doc[token.i].is_sent_start = True
        return doc

    @classmethod
    def load_big_object(cls, _name):
        nlp_model = spacy.load('en_core_web_sm', disable=["ner", "textcat"])
        nlp_model.add_pipe('newline_sentencizer', before="parser")
        for key in EMOTICONS_EMO:
            nlp_model.tokenizer.add_special_case(key, [{"ORTH": key}])
        return nlp_model


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info('Start spaCy preprocessing...')
        with GPUManager.gpu_routine(spacy.require_gpu, spacy.require_cpu):
            docs = list(self.nlp.pipe(X, batch_size=5000, n_process=1))
        logger.info('SpaCy preprocessing finished')
        return docs
