""" Usage:
trained_oie_extractor --model=MODEL_DIR --in=INPUT_FILE --out=OUTPUT_FILE

Run a trined OIE model on tokenized sentences.

MODEL_DIR - Pretrained RNN model folder (containing model.json and pretrained weights)
INPUT FILE - File where each row is a tokenized sentence to be parsed with OIE
OUTPUT_FILE - File where the OIE tuples will be output.

TODO: specify format of OUTPUT_FILE
"""

from rnn.model import load_pretrained_rnn
from docopt import docopt
import logging

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


if __name__ == "__main__":
    args = docopt(__doc__)
    logging.debug(args)
    model_dir = args["--model"]
    input_fn = args["--in"]
    output_fn = args["--out"]

    oie = Trained_oie(load_pretrained_rnn(model_dir))
