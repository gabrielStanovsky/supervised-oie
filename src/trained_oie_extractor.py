""" Usage:
trained_oie_extractor --model=MODEL_DIR --in=INPUT_FILE --out=OUTPUT_FILE [--tokenize]

Run a trined OIE model on raw sentences.

MODEL_DIR - Pretrained RNN model folder (containing model.json and pretrained weights).
INPUT FILE - File where each row is a tokenized sentence to be parsed with OIE.
OUTPUT_FILE - File where the OIE tuples will be output.
tokenize - indicates that the input sentences are NOT tokenized.

TODO: specify format of OUTPUT_FILE
"""

from rnn.model import load_pretrained_rnn
from docopt import docopt
import logging
import nltk
import numpy as np

logging.basicConfig(level = logging.DEBUG)

class Trained_oie:
    """
    Compose OIE extractions given a pretrained RNN OIE model predicting classes per word
    """
    def __init__(self, model):
        """
        model - pretrained supervised model
        """
        self.model = model

    def get_extractions(self, sent):
        """
        Returns a list of OIE extractions for a given sentence
        sent - a list of tokens
        """
        for ((pred_ind, pred_word), labels) self.model.predict_sentence(sent):
            cur_args = []
            cur_arg = []
            cur_prob = 1.0
            # collect args
            for word_ind, (label, prob) in enumerate()

    def parse_sents(self, sents, tokenize):
        """
        Returns a list of extractions per sent in sents.
        sents - list of tokenized sentences
        tokenize - boolean indicating whether the sentences should be tokenized first
        """
        return [self.get_extractions(nltk.word_tokenize(sent) if tokenize else sent)
                for sent in sents]


class Extraction:
    """
    Store and print an OIE extraction
    """
    def __init__(self, sent, prob, pred, argsx):
        """
        sent - Tokenized sentence - list of strings
        pred - Predicate word
        args - List of arguments (each a string)
        prob - Gloat in [0,1] indicating the probablity
               of this extraction
        """
        self.sent = sent
        self.prob = prob
        self.pred = pred
        self.args = args

    def __str__(self):
        """
        Format (tab separated):
        Sent, prob, pred, arg1, arg2, ...
        """
        return '\t'.join(map(str,
                             [' '.join(self.sent),
                              self.prob,
                              self.pred,
                              '\t'.join(self.args)]))

if __name__ == "__main__":
    args = docopt(__doc__)
    logging.debug(args)
    model_dir = args["--model"]
    input_fn = args["--in"]
    output_fn = args["--out"]
    tokenize = args["--tokenize"]

    oie = Trained_oie(load_pretrained_rnn(model_dir))

    s = "The Economist is an English language weekly magazine format newspaper owned by the Economist Group\
    and edited at offices in London."
    y = oie.parse_sents([s], tokenize = True)
