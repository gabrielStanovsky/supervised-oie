""" Usage:
    convert_from_mesquita --in=INPUT_FILE --out=OUTPUT_FILE [--verbal] [--debug]

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
from parsers.spacy_wrapper import spacy_whitespace_parser as spacy_ws

#----


def find_enclosed_elem(annotated_sent, start_symbol, end_symbol):
    """
    Extract elements enclosed by some denotation.
    """
    sexp = nestedExpr(start_symbol,
                      end_symbol,
                      ignoreExpr = None).parseString("{}{}{}".format(start_symbol,
                                                                     annotated_sent,
                                                                     end_symbol)).asList()[0]
    exps = [get_raw_sent(" ".join(ls))
            for ls in sexp
            if isinstance(ls, list)]

    # Make sure there's a single predicate head
    return exps


def get_entities(annotated_sent):
    """
    Get the entities participating in this sentence.
    """
    return [elem.split(' ', 1)[1] # Drop the NER label
            for elem
            in find_enclosed_elem(annotated_sent,
                                  '[[[',
                                  ']]]')]

def get_predicate_head(annotated_sent):
    """
    Get the predicate head annotated in the input.
    """
    preds = find_enclosed_elem(annotated_sent,
                               '{{{',
                               '}}}')

    if not preds:
        return []
    # Make sure there's a single predicate head
    assert(len(preds) == 1)
    return preds[0]


def get_predicate(annotated_sent):
    """
    Get the predicate in this sentence.
    """
    pred_start = "--->"
    pred_end = "<---"

    # Some predicates are non-contiguous
    return " ".join(find_enclosed_elem(annotated_sent,
                                       "--->",
                                       "<---"))

SPECIAL_CHARS = ["{{{",
                 "}}}",
                 "]]]",
                 " --->",
                 "<--- ",
                 "--->", # Should appear after their super-string
                 "<---",
                 "-->"] # Probably a bug in the original annotation

def get_raw_sent(annotated_sent):
    """
    Return the raw sentence, without the special characters.
    """
    ret = annotated_sent
    for chars in SPECIAL_CHARS:
        ret = ret.replace(chars, '')

    return " ".join([word for word
                     in ret.split(' ')
                     if not word.startswith("[[[")]).strip()

def strip_word_index(line):
    """
    Remove the word indices from string.
    """
    try:
        return " ".join([word.split('_', 1)[1]
                         for word in line.split(' ')])
    except:
        pdb.set_trace()

def convert_single_sent(annotated_sent, verbal):
    """
    Return our format for a single annotated sentence.
    Verbal controls whether only verbal extractions should be made.
    From Mesquita's readme:
    Annotated Sentence:  The sentence annotated with the entity pair, the trigger and allowed tokens.
                        Entities are enclosed in triple square brackets, triggers are enclosed in
                        triple curly brackets and the allowed tokens are enclosed in arrows.
                        ("--->" and "<---").
    """
    proc_sent = []
    word_ind = 0
    for word in annotated_sent.split():
        if (word not in SPECIAL_CHARS):
            if "{{{" in word:
                # Boilerplate index
                bp_ind = word.index('{{{') + 3
                # Plant the index in the correct place
                word = "{}{}_{}".format(word[0 : bp_ind],
                                        word_ind,
                                        word[bp_ind :])
                word_ind += 1

            elif not (word.startswith("[[[")):
                word = "{}_{}".format(word_ind, word)
                word_ind += 1

        proc_sent.append(word)

    proc_sent = " ".join(proc_sent)
    pred = get_predicate_head(proc_sent)
    raw_sent = get_raw_sent(proc_sent)
    doc = spacy_ws(strip_word_index(raw_sent))

    # Filter non-verbs and empty predicates
    if (not pred) or \
       (verbal and \
        (not doc[int(pred.split("_")[0])].tag_.startswith("V"))):
        return None
    return map(strip_word_index,
               [raw_sent, pred] + get_entities(proc_sent))


if __name__ == "__main__":

    # Parse command line arguments
    args = docopt(__doc__)
    inp_fn = args["--in"]
    out_fn = args["--out"]
    verbal = args["--verbal"]

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

    ls = []
    for annotated_sent in inp_df["Annotated Sentence"].values:
        conv_sent = convert_single_sent(annotated_sent, verbal)
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
