""" Usage:
    load_pretrained_word_embeddings [--glove=GLOVE_FN]
"""

from docopt import docopt
import numpy as np
from word_index import Word_index
import logging
logging.basicConfig(level = logging.DEBUG)
import sys
sys.path.append("./common")
from symbols import UNK_INDEX, UNK_SYMBOL, UNK_VALUE
from keras .layers import Embedding

class Glove:
    """
    Stores pretrained word embeddings for GloVe, and
    outputs a Keras Embeddings layer.
    """
    def __init__(self, fn, dim = None):
        """
        Load a GloVe pretrained embeddings model.
        fn - Filename from which to load the embeddings
        dim - Dimension of expected word embeddings, used as verficiation,
              None avoids this check.
        """
        self.fn = fn
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self._load(self.fn)
        logging.debug("Done!")

    def _load(self, fn):
        """
        Load glove embedding from a given filename
        """
        self.word_index = {UNK_SYMBOL : UNK_INDEX}
        emb = []
        for line in open(fn):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if self.dim:
                assert(len(coefs) == self.dim)
            else:
                self.dim = len(coefs)

            # Record mapping from word to index
            self.word_index[word] = len(emb) + 1
            emb.append(coefs)

        # Add UNK at the first index in the table
        self.emb = np.array([UNK_VALUE(self.dim)] + emb)
        # Set the vobabulary size
        self.vocab_size = len(self.emb)

    def get_word_index(self, word, lower = True):
        """
        Get the index of a given word (int).
        If word doesnt exists, returns UNK.
        lower - controls whether the word should be lowered before checking map
        """
        if lower:
            word = word.lower()
        return self.word_index[word] \
            if (word in self.word_index) else UNK_INDEX

    def get_embedding_matrix(self):
        """
        Return an embedding matrix for use in a Keras Embeddding layer
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        word_index - Maps words in the dictionary to their index (one-hot encoding)
        """
        return self.emb

    def get_keras_embedding(self, **args):
        """
        Get a Keras Embedding layer, loading this embedding as pretrained weights
        The additional arguments given to this function are passed to the Keras Embbeding constructor.
        """
        return Embedding(self.vocab_size,
                         self.dim,
                         weights = [self.get_embedding_matrix()],
                         **args)



if __name__ == "__main__":
    args = docopt(__doc__)
    if "--glove" in args:
        glove_fn = args["--glove"]
        emb = Glove(glove_fn)
        mat = emb.get_embedding_matrix()
        logging.debug("Emb size: {}".format(emb.vocab_size))
    else:
        logging.info(__doc__)
        exit
