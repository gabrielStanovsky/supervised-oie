"""
Functions to mediate the spaCy interface.
"""

import spacy
from spacy.tokens import Doc
from operator import itemgetter
import re
import logging
import pdb


class WhitespaceTokenizer(object):
    """
    White space tokenizer - assumes all text space-separated
    https://spacy.io/docs/usage/customizing-tokenizer
    """
    def __init__(self, nlp):
        """
        Get an initialized spacy object
        """
        self.vocab = nlp.vocab

    def __call__(self, text):
        """
        Call this tokenizer - just split based on space
        """
        words = re.split(r' +', text) # Allow arbitrary number of spaces
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)



def set_sent_starts(doc):
    # Mark each doc as a single sentence.
    sent_start_char = 0
    sent_end_char = len(doc.text)
    sent = doc.char_span(sent_start_char, sent_end_char)
    sent[0].sent_start = True
    for token in sent[1:]:
        token.sent_start = False
    return doc


class spacy_with_whitespace_tokenizer:
    """
    Get a spacy parser instance with white space tokenization.
    """
    parser = spacy.load('en',
                        create_make_doc = WhitespaceTokenizer)

    # TODO: migrate to spacy V2.0, make sure that whitespacetokenizer still
    #       works
    # parser.add_pipe(set_sent_starts,
    #                 name='sentence_segmenter', before='parser')


def get_conll(doc):
    """
    Get CONLL string representation of the parsed
    spacy instance.
    https://github.com/explosion/spaCy/issues/533
    """
    ret = []
    for sent in doc.sents:
        for i, word in enumerate(sent):
            if word.head is word:
                head_idx = 0
            else:
                head_idx = doc[i].head.i+1

            ret.append("%d\t%s\t%s\t%s\t%s\t%s\t%d\t%s\t%s\t%s"%(
                i+1, # There's a word.i attr that's position in *doc*
                word,
                '_',
                word.pos_, # Coarse-grained tag
                word.tag_, # Fine-grained tag
                '_',
                head_idx,
                word.dep_, # Relation
                '_', '_'))

    return "\n".join(ret)


def spacy_whitespace_parser(text, encoding = "utf8"):
    """
    Parse sentence with static instance.
    """
#    logging.debug("spacy text: {}".format(text))
    return spacy_with_whitespace_tokenizer.parser(unicode(text,
                                                          encoding))
