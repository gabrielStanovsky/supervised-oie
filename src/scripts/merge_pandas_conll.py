""" Usage:
    merge_pandas_conll --out=OUTPUT_FN <filenames>...

Merge a list of data frames in csv format and print to output file.
"""
from docopt import docopt
import pandas as pd
import logging
logging.basicConfig(level = logging.DEBUG)

if __name__ == "__main__":
    args = docopt(__doc__)
    logging.debug(args)
    input_fns = args["<filenames>"]
    out_fn = args["--out"]
    pd.concat([pd.read_csv(fn,
                           sep = '\t',
                           header = 0)
               for fn in input_fns]).to_csv(out_fn,
                                            sep = '\t',
                                            header = True,
                                            index = False)
