""" Usage:
   label_frequency --in=INPUT_FN --out=OUTPUT_FN

Compute the label frequency of a given input file and print to a csv format to output file
"""
from collections import defaultdict
from docopt import docopt
import logging
logging.basicConfig(level = logging.DEBUG)
import sys
sys.path.append("./common")
from utils import joinstr
import pandas

if __name__ == "__main__":
    args = docopt(__doc__)
    input_file = args["--in"]
    output_file = args["--out"]

    df = pandas.read_csv(input_file, sep = '\t', header = 0)
    labels_dic = dict([(label, df[df.label == label].shape[0]) for
                       label in df.label.unique()])
    total = df.label.shape[0]

    # Print to output file
    with open(output_file, 'w') as fout:
        for label, count in sorted(labels_dic.iteritems(), key = lambda (k, v): v, reverse = True):
            fout.write(joinstr(',', [label, count, '{:.3f}'.format((float(count) / total)*100)]) + '\n')
