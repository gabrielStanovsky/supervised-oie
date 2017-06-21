""" Usage:
    seq2seq_model --train=TRAIN_FN --dev=DEV_FN --test=TEST_FN --hyperparams=MODEL_JSON --saveto=MODEL_DIR [-v]

Train a seq2seq OIE model.

Parameters:
  train                The train file
  dev                  The development file
  test                 The test file
  hyperparams          Json paramters to init model
  saveto               Where to store the trained model
  v                    Verbose
"""

import seq2seq
from keras.utils import plot_model
from seq2seq.models import Seq2Seq, AttentionSeq2Seq
from docopt import docopt
from pprint import pformat
import logging
import json

from load_pretrained_word_embeddings import Glove

class Seq2seq_OIE:
    """
    Modelling OIE as sequence to sequence, allows for introducing non-sentenial words into
    the predictions, and not having to align the tuples words.
    """
    def __init__(self,
                 batch_size,
                 maximum_output_length,
                 emb_fn,
                 hidden_dim,
                 input_depth,
                 output_depth,
                 peek,
                 attention,
                 seed,
    ):
        """
        Init and compile model's params
        Arguments:
        batch_size - (=input_lenght) Batch size in which to partition the elements
        maximum_output_length - The maximum number of words in output
        emb - Pretrained embeddings
        hidden_dim - number of hidden units
        input_depth - the number of layers in encoder
        output_depth - the number of layers in decoder
        peek - (binray) add the peek feature
        attention - (binary) use attention model
        """
        self.emb = Glove(emb_fn)
        self.model = Seq2seq_OIE.compile_model(input_length = batch_size,
                                               input_depth = input_depth,
                                               input_dim = self.emb.dim,
                                               hidden_dim = hidden_dim,
                                               output_length = maximum_output_length,
                                               output_depth = output_depth,
                                               output_dim = self.emb.dim,
                                               peek = peek,
                                               attention = attention,
        )


    @staticmethod
    def compile_model(input_length,
                      input_depth,
                      input_dim,
                      hidden_dim,
                      output_length,
                      output_depth,
                      output_dim,
                      peek,
                      attention,
    ):
        """
        Returns a compiled seq2seq model
        Arguments:
        input_length - Batch size in which to partition the elements
        input_depth - the number of layers in encoder
        input_dim - the number of features for each word
        hidden_dim - number of hidden units
        output_length - (= maximum_output_length) The maximum number of words in output
        output_depth - the number of layers in decoder
        output_dim - the number of features in word embedding's output
        peek - (binray) add the peek feature
        attention - (binary) use attention model
        """
        model_fn = AttentionSeq2Seq \
                   if attention \
                      else Seq2Seq

        model = model_fn(input_length = input_length,
                         input_dim = input_dim,
                         hidden_dim = hidden_dim,
                         output_length = output_length,
                         output_dim = output_dim,
                         depth = (input_depth,
                                  output_depth),
#                         peek = peek, TODO
        )
        model.compile(loss='mse', optimizer='rmsprop')
        if logging.getLogger().getEffectiveLevel() == logging.DEBUG:
            model.summary()
        return model


if __name__ == "__main__":
    # Parse arguments
    args = docopt(__doc__)
    verbosity_level = logging.DEBUG if args['-v']\
                      else logging.INFO
    logging.basicConfig(level = verbosity_level)

    logging.debug(pformat(args))
    train_fn = args['--train']
    dev_fn = args['--dev']
    test_fn = args['--test']
    hyperparams_fn = args['--hyperparams']
    output_fn = args['--saveto']

    # Parse hyperparams and initialize model
    hyperparams = json.load(open(hyperparams_fn))['hyperparams']
    logging.debug("Model hyperparms: {}".format(pformat(hyperparams)))
    s2s = Seq2seq_OIE(**hyperparams)
