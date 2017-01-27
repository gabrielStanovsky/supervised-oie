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
        ret = []
        for ((pred_ind, pred_word), labels) in self.model.predict_sentence(sent):
            cur_args = []
            cur_arg = []
            cur_prob = 1.0

            # collect args
            for (label, prob), word in zip(labels, sent):
                if label.startswith("A"):
                    cur_arg.append(word)
                    if prob < cur_prob:
                        cur_prob = prob
#                    cur_prob *= prob
                elif cur_arg:
                    cur_args.append(cur_arg)
                    cur_arg = []

            # Create extraction
            if cur_args:
                ret.append(Extraction(sent,
                                      cur_prob,
                                      pred_word,
                                      cur_args))
        return ret

    def parse_sent(self, sent, tokenize):
        """
        Returns a list of extractions for the given sentence
        sent - a tokenized sentence
        tokenize - boolean indicating whether the sentences should be tokenized first
        """
        return self.get_extractions(nltk.word_tokenize(sent) if tokenize else sent)

    def parse_sents(self, sents, tokenize):
        """
        Returns a list of extractions per sent in sents.
        sents - list of tokenized sentences
        tokenize - boolean indicating whether the sentences should be tokenized first
        """
        return [self.parse_sent(sent, tokenize)
                for sent in sents]


class Extraction:
    """
    Store and print an OIE extraction
    """
    def __init__(self, sent, prob, pred, args):
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
        logging.debug(self)

    def __str__(self):
        """
        Format (tab separated):
        Sent, prob, pred, arg1, arg2, ...
        """
        return '\t'.join(map(str,
                             [' '.join(self.sent),
                              self.prob,
                              self.pred,
                              '\t'.join([' '.join(arg)
                                         for arg in self.args])]))


example_sent = "The Economist is an English language weekly magazine format newspaper owned by the Economist Group\
    and edited at offices in London."


if __name__ == "__main__":
    args = docopt(__doc__)
    logging.debug(args)
    model_dir = args["--model"]
    input_fn = args["--in"]
    output_fn = args["--out"]
    tokenize = args["--tokenize"]

    oie = Trained_oie(load_pretrained_rnn(model_dir))

    # Iterate over all raw sentences
    with open(output_fn, 'w') as fout:
        fout.write('\n'.join([str(ex)
                              for sent in open(input_fn)
                              for ex in oie.parse_sent(sent.strip().split(' '),
                                                       tokenize = tokenize)
                              if sent.strip()]))
