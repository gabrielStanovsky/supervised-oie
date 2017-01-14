""" Usage:
   relabel --in=INPUT_FN --out=OUTPUT_FN

Relabel an OIE file to get a better distribution:
Adds a posfix label to the Outside tokens - with the before and after arguments
The start and end section also receive a different postfix (S, E)
"""
from collections import defaultdict
from docopt import docopt
import logging
logging.basicConfig(level = logging.DEBUG)
import sys
sys.path.append("./common")
from utils import joinstr
import pandas
from pprint import pprint

def relabel(sent):
    """
    Adds a posfix label to the Outside tokens - with the before and after arguments
    """
    # Pandas wasn't nice for changing the values of df

    postfix = 'S'
    ret = []

    words = list(sent.iterrows())

    for _, word in words:
        if word.label == "O":
            new_label = "O-{}".format(postfix)
        else:
            if all([next_word.label == 'O' for (_, next_word)
                    in words[word.word_id + 1 : ]]):
                # TODO: this is very inefficient
                postfix = 'E'
            else:
                postfix = word.label.split("-")[0] # Borrow label from last seen tag
            new_label = word.label

        # Reform line, only label is possibly changed
        ret.append([word.word_id, word.word, word.pred, word.pred_id, word.sent_id, word.run_id,
                    new_label])
    return ret

if __name__ == "__main__":
    args = docopt(__doc__)
    input_file = args["--in"]
    output_file = args["--out"]

    # iterate over sentences
    df = pandas.read_csv(input_file, sep = '\t', header = 0)
    sents = [df[df.run_id == i] for i in range(min(df.run_id), max(df.run_id))]

    with open(output_file, 'w') as fout:
        # Write header
        fout.write(joinstr('\t', [k for k in df.keys()]))
        #Write sents:
        for sent in sents:
            for line in relabel(sent):
                fout.write(joinstr('\t', line))
            fout.write('\n')
