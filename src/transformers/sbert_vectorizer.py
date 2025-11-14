import gc
import logging
import re
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sklearn.base import BaseEstimator, TransformerMixin
from transformers import logging as hf_logging
from src.util import GPUManager, PickleCompatible

logger = logging.getLogger(__name__)

class SbertVectorizer(BaseEstimator, TransformerMixin, GPUManager, PickleCompatible):
    _big_objects = ['model', 'tokenizer']
    _model_name = 'sentence-transformers/all-mpnet-base-v2'

    def __init__(self):
        self.chunk_token_size = self.model.max_seq_length - 50
        logger.info('Max seq length: %i', self.chunk_token_size)
        self.overlap = int(self.chunk_token_size * 0.2)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_token_size,
            chunk_overlap=self.overlap,
            length_function=self._token_length,
            # Punctuation without spaces. Could be a sign of censorship
            separators=["\n\n", "\n", ",", " ", "!", ".", "?", "'"]
        )

    @classmethod
    def load_big_object(cls, name):
        match name:
            case 'model':
                return SentenceTransformer(cls._model_name, device=GPUManager.device())
            case 'tokenizer':
                return cls.lazy_load('model').tokenizer

    def _token_length(self, text):
        return len(self.tokenizer.encode(text, add_special_tokens=True))

    def _chunk_text_by_tokens(self, text):
        # Token indices sequence length is longer...
        # Disable this message here because we're not going to run this sequence through the model
        logging_level = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        n_tokens = self._token_length(text)
        sentences = None
        if n_tokens > self.chunk_token_size:
            sentences = re.split(r'(?<=[.!?])\s+', text)
        else:
            return [text]
        chunks = []
        for sent in sentences:
            chunks.extend(self.splitter.split_text(sent))
        self.tokenizer.deprecation_warnings.pop(
            "sequence-length-is-longer-than-the-specified-maximum",
            None
        )
        # And enable logging again because we want to know if there is a long chunk
        hf_logging.set_verbosity(logging_level)
        return chunks

    def _agg_embeddings(self, chunks, embeddings):
        lengths = np.array([len(c) for c in chunks], dtype=float)
        weights = lengths / lengths.sum()
        return (embeddings * weights[:, None]).sum(axis=0)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        logger.info('Start SBERT transform')
        X_encoded = []
        X = X if isinstance(X, list) else list(X)
        with GPUManager.gpu_routine(lambda: self.model.to(GPUManager.device()), self.model.cpu):
            for idx, x in enumerate(X):
                chunks = self._chunk_text_by_tokens(x)
                if not chunks:
                    emb_dim = self.model.get_sentence_embedding_dimension()
                    X_encoded.append(np.zeros(emb_dim))
                    continue

                chunk_embeddings = self.model.encode(
                    chunks,
                    device=GPUManager.device(),
                    batch_size=32,
                    convert_to_numpy=True,
                    show_progress_bar=False
                )
                wegihted_embeddings = self._agg_embeddings(chunks, chunk_embeddings)
                X_encoded.append(wegihted_embeddings)
                if (idx + 1) % 500 == 0:
                    torch.cuda.empty_cache()
                    gc.collect()
        logger.info('Finish SBERT transform')
        return np.vstack(X_encoded)
