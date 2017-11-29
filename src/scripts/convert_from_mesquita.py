""" Usage:
    convert_from_mesquita --in=INPUT_FILE --out=OUTPUT_FILE [--debug]

Convert ground truth annotation from Mesquita et al. to our format.

"""
# External imports
import logging
import pdb
import pandas as pd
from pprint import pprint
from pprint import pformat
from docopt import docopt
from pyparsing import nestedExpr

# Local imports


#----


def get_entities(annotated_sent):
    """
    Get the entities participating in this sentence.
    """
    sexp = nestedExpr('[[[',
                      ']]]',
                      ignoreExpr = None).parseString("[[[{}]]]".format(annotated_sent)).asList()[0]
    return [" ".join(ls[1:])  # Drop the NER label
            for ls in sexp
            if isinstance(ls, list)]

def get_predicate(annotated_sent):
    """
    Get the predicate in this sentence.
    """
    if "--->" in annotated_sent:
        pred_start = "--->"
        pred_end = "<---"
    else:
        # If there's no allowed word marks, try just the bare predicate
        pred_start = "--->"
        pred_end = "<---"

        # pred_start = "{{{"
        # pred_end = "}}}"

    sexp = nestedExpr(pred_start,
                      pred_end,
                      ignoreExpr = None).parseString("{}{}{}".format(pred_start,
                                                                     annotated_sent,
                                                                     pred_end)).asList()[0]
    preds = [get_raw_sent(" ".join(ls))
             for ls in sexp
             if isinstance(ls, list)]

    if not preds:
        #TODO: Some entries don't annotate a predicate?
        return None

    return " ".join(preds) # Some predicates are non-contiguous


def get_raw_sent(annotated_sent):
    """
    Return the raw sentence, without the special characters.
    """
    ret = annotated_sent.replace("{{{", ''). \
          replace("}}}", ''). \
          replace("]]]", ''). \
          replace(" --->", ''). \
          replace("<--- ", '')

    return " ".join([word for word
                     in ret.split(' ')
                     if not word.startswith("[[[")])



def convert_single_sent(annotated_sent):
    """
    Return our format for a single annotated sentence.
    From Mesquita's readme:
    Annotated Sentence:  The sentence annotated with the entity pair, the trigger and allowed tokens.
                        Entities are enclosed in triple square brackets, triggers are enclosed in
                        triple curly brackets and the allowed tokens are enclosed in arrows.
                        ("--->" and "<---").
    """
    logging.debug(annotated_sent)
    pred = get_predicate(annotated_sent)
    if pred is None:
        return None
    return [get_raw_sent(annotated_sent),
            pred] + get_entities(annotated_sent)


if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]

    # Determine logging level
    debug = args["--debug"]
    if debug:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)

    ## Start computation
    # Parse input df
    inp_df = pd.read_csv(inp_fn,
                         sep = '\t',
                         header = 0)

    # Create output df and write to file
    # out_df = pd.DataFrame([pd.Series(convert_single_sent(annotated_sent))
    #                        for annotated_sent

    ls = []
    for annotated_sent in inp_df["Annotated Sentence"].values:
        conv_sent = convert_single_sent(annotated_sent)
        if conv_sent:
            # Ignoring annotations without a predicate
            ls.append(pd.Series(conv_sent))

    out_df = pd.DataFrame(ls)

    logging.info("Writing {} tuples to {}".format(len(out_df),
                                                  out_fn))

    out_df.to_csv(out_fn,
                  sep = '\t',
                  header = None,
                  index = None)

    # End
    logging.info("DONE")
