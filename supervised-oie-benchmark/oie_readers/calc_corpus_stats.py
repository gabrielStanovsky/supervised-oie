""" Usage:
    calc_corpus_stats (--gold=INPUT_FILENAME| --in=INPUT_FILENAME) --out=OUTPUT_FILENAME

Prints various stats about a given corpus to the output file. The input file should be an Open IE corpus in tabbed
format.

Stats prints:

1. Number of extractions
1. Average length of extractions
"""
from tabReader import TabReader
from goldReader import GoldReader
from docopt import docopt
import numpy as np
import logging
logging.basicConfig(level = logging.DEBUG)


if __name__ == "__main__":
    args = docopt(__doc__)
    out = args["--out"]
    if args["--in"] is not None:
        oie = TabReader()

    elif args["--gold"] is not None:
        oie = GoldReader()

    inp = args["--in"] or args["--gold"]
    oie.read(inp)
    exs = [ex for exts_list in oie.oie.values() for ex in exts_list]
    with open(out, 'w') as fout:
        num_of_extractions = len(exs)

        logging.debug("# Extractions: {}\n".\
                      format(num_of_extractions))

        fout.write("# Extractions: {}\n".\
                   format(num_of_extractions))

        average_num_of_args = np.average([len(ex.args)
                                        for ex in exs])

        logging.debug("Average # arguments / extraction: {}\n".\
                  format(average_num_of_args))

        fout.write("Average # arguments / extraction: {:.2f}\n".\
                  format(average_num_of_args))

        average_num_of_words_in_args = np.average([len(arg) for
                                     ex in exs
                                     for arg in ex.args])
        logging.debug("Average # words / argument: {}\n".\
                  format(average_num_of_words_in_args))

        fout.write("Average # words / argument: {:.2f}\n".\
                  format(average_num_of_words_in_args))

        average_pred_len = np.average([len(ex.pred.split(" ")) for
                                                    ex in exs])
        logging.debug("average predicate length: {}\n".\
                  format(average_pred_len))

        fout.write("average predicate length: {:2f}\n".\
                  format(average_pred_len))

        average_prop_len = np.average([len(ex.pred.split(" ")) +
                                       sum([len(arg) for arg in ex.args])
                                       for ex in exs])
        logging.debug("average prop length: {}\n".\
                  format(average_prop_len))

        fout.write("average prop length: {:2f}\n".\
                  format(average_prop_len))
