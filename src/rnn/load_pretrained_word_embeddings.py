""" Usage:
    load_pretrained_word_embeddings [--glove=GLOVE_FN]
"""

from docopt import docopt
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)

class Glove:
    """
    Stores pretrained word embeddings for GloVe, and
    outputs a Keras Embeddings layer.
    """
    def __init__(self, fn, nb_words = None, dim = None):
        """
        Load a GloVe pretrained embeddings model.
        fn - Filename from which to load the embeddings
        nb_words - Maximum number of words to include (None includes all words)
        dim - Dimension of expected word embeddings, used as verficiation,
              None avoids this check.
        """
        self.fn = fn
        self.nb_words = nb_words
        self.dim = dim
        logging.debug("Loading GloVe embeddings from: {} ...".format(self.fn))
        self.embedding_index = self._load_glove(self.fn)
        logging.debug("Done!")

    def _load_glove(self, fn):
        """
        Load glove embedding from a given filename
        """
        ret = {}
        for line in open(fn):
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            if self.dim:
                assert(len(coefs) == self.dim)
            else:
                self.dim = len(coefs)
            ret[word] = coefs
        return ret

    def get_embedding_matrix(self, word_index):
        """
        Return an embedding matrix for use in a Keras Embeddding layer
        https://blog.keras.io/using-pre-trained-word-embeddings-in-a-keras-model.html
        word_index - Maps words in the dictionary to their index (one-hot encoding)
        """
        self.embedding_matrix = np.zeros((len(word_index) + 2, self.dim)) # Word indices start from 1 and have an UNK symbol
        for word, i in word_index.iteritems():
            logging.debug("Adding {}".format((word, i)))
            embedding_vector = self.embedding_index.get(word)
            if embedding_vector is not None:
                # Words not found in embedding index will be all-zeros.
                # Note - but these words will still have their own vector, not unkified
                self.embedding_matrix[i] = embedding_vector

        return self.embedding_matrix

if __name__ == "__main__":
    args = docopt(__doc__)
    if "--glove" in args:
        glove_fn = args["--glove"]
        emb = Glove(glove_fn)
        word_index = dict(zip("my dog ate the home work".split(" "),
                                                range(1,7)))
        mat = emb.get_embedding_matrix(word_index)
    else:
        logging.info(__doc__)
        exit
