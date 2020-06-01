import math
import logging
import os
import io

from torch import nn, Tensor, from_numpy
import numpy as np
from joeynmt.helpers import freeze_params
from joeynmt.vocabulary import Vocabulary

logging.root.setLevel(logging.INFO)

class Embeddings(nn.Module):

    """
    Simple embeddings class
    """

    # pylint: disable=unused-argument
    def __init__(self,
                 embedding_dim: int = 64,
                 scale: bool = False,
                 vocab_size: int = 0,
                 padding_idx: int = 1,
                 freeze: bool = False,
                 **kwargs):
        """
        Create new embeddings for the vocabulary.
        Use scaling for the Transformer.

        :param embedding_dim:
        :param scale:
        :param vocab_size:
        :param padding_idx:
        :param freeze: freeze the embeddings during training
        """
        super(Embeddings, self).__init__()

        self.embedding_dim = embedding_dim
        self.scale = scale
        self.vocab_size = vocab_size
        self.lut = nn.Embedding(vocab_size, self.embedding_dim,
                                padding_idx=padding_idx)

        if freeze:
            freeze_params(self)

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor) -> Tensor:
        """
        Perform lookup for input `x` in the embedding table.

        :param x: index in the vocabulary
        :return: embedded representation for `x`
        """
        if self.scale:
            return self.lut(x) * math.sqrt(self.embedding_dim)
        return self.lut(x)

    def __repr__(self):
        return "%s(embedding_dim=%d, vocab_size=%d)" % (
            self.__class__.__name__, self.embedding_dim, self.vocab_size)


def reload_txt_emb(path: str, embedding_dim: int = 300) -> tuple:
    """
    Reload pretrained embeddings from a text file.
    """
    assert os.path.isfile(path) and embedding_dim > 0
    word2id = {}
    vectors = []
    logging.info("Reloading embeddings from %s ..." % path)

    with io.open(
        path, 'r', encoding='utf-8', newline='\n', errors='ignore'
        ) as f:
        next(f)
        for i, line in enumerate(f):
            word, vect = line.rstrip().split(' ', 1)
            vect = np.fromstring(vect, sep=' ').astype(np.float32)
            assert word not in word2id, 'word found twice'
            assert vect.shape == (embedding_dim,), i
            vectors.append(vect[None])
            word2id[word] = len(word2id)

    vectors = np.concatenate(vectors, 0)
    logging.info("Reloaded %i embeddings with dimension %i" % (
        len(vectors), vectors.shape)
        )
    return vectors, word2id

def initialize_embeddings(config: dict, vocab: Vocabulary, embeddings: Embeddings):
    """
    Initialize embeddings from pretrained weights
    """
    pretrained_path = config.get("pretrained_embed_path", None)
    pretrained_dim = config.get("pretrained_embed_dim", None)
    
    pretrained, word2id = reload_txt_emb(pretrained_path, pretrained_dim)
    found = 0
    lower = 0

    for word_id in range(len(vocab.itos)):
        word = vocab.itos[word_id]
        if word in word2id:
            found += 1
            vec = from_numpy(pretrained[word2id[word]])
            embeddings.lut.weight.data[word_id] = vec
        elif word.lower() in word2id:
            found += 1
            lower += 1
            vec = from_numpy(pretrained[word2id[word.lower()]])
            embeddings.lut.weight.data[word_id] = vec

    logging.info(
            "Initialized %i word embeddings, including %i "
            "after lowercasing." % (found, lower)
        )
