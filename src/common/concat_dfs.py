""" Usage:
    concat_dfs --df1=DF1_FILE --df2=DF2_FILE --out=OUTPUT_FILE [--debug]
"""
# External imports
import logging
import pdb
import pandas as pd
from pprint import pprint
from pprint import pformat
from docopt import docopt

# Local imports
from utils import concat_dfs
from utils import df_to_conll

#----

if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    df1_fn = args["--df1"]
    df2_fn = args["--df2"]
    out_fn = args["--out"]

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    # Start computation
    df1 = pd.read_csv(df1_fn,
                      sep = '\t',
                      header = 0,
                      keep_default_na = False,
                      na_filter = False)

    df2 = pd.read_csv(df2_fn,
                      sep = '\t',
                      header = 0,
                      keep_default_na = False,
                      na_filter = False)

    logging.info("Writing concatenation to {}".format(out_fn))
    out_df = concat_dfs(df1,
                           df2,
                           running_keys = ["sent_id",
                                           "run_id"])
    df_to_conll(out_df,
                out_fn)

    # End
    logging.info("DONE")
