"""
Functions to mediate the spaCy interface.
"""

import spacy
from spacy.tokens import Doc
from util.preproc import enum
from operator import itemgetter
import logging
import pdb


## Enum class for representing a chunk's side, relative
## to the head.
Sides = enum(LEFT = -1,
             MIDDLE = 0,
             RIGHT = 1)

class Chunk:
    """
    Container for spacy chunks
    """
    def __init__(self, head, toks, side):
        """
        head - the head of this chunk
        toks - the tokens participating in this chunk (including head)
        side - (Sides) whether this chunk is left or right of the head
        """
        self.head = head
        self.toks = sorted(list(set([head] + list(toks))),
                           key = lambda tok: tok.i)
        self.side = side

        # For sorting purposes, assign this chunk's
        # head position as its index, can be further manipulated
        # from outside the object's functions.
        self.chunk_index = self.head.i

    def is_wh_chunk(self):
        """
        Returns True iff this chunk has a WH token.
        """
        return (len(self.toks) == 1) \
            and is_wh(self.toks[0])

    def is_mapped(self, qa):
        """
        Check whether this chunk is mapped to the sentence
        in a given qa pair.
        """
        return qa.aligned_question[self.head.i].is_mapped

    def __str__(self):
        """
        Raw textual linerization of this chunk.
        """
        return " ".join([str(tok)
                         for tok in self.toks])

class ParsedQuestion:
    """
    Chunking and structuring a question phrase.
    """
    def __init__(self, head_vp):
        """
        head_vp - the head Chunk of the question
        """
        self.root = head_vp
        self.children = []

    def add_child_chunk(self, rel, child_chunk):
        """
        Add a child chunk with a given textual relation
        """
        self.children.append((rel, child_chunk))

    def find_wh_chunks(self):
        """
        Find wh relations within this question.
        Returns:
        (relation to the head (the first relation in the path from the head),
         side with relation to the head)
        """
        return [(rel, chunk, chunk.side)
                for (rel, chunk) in self.children
                if chunk.is_wh_chunk()]

    def _fix_wh_location(self, sorted_args):
        """
        Place the wh chunk in the correct linear position
        """
        ret = []

        wh_index, wh_rel, wh_chunk = [(i, rel, chunk)
                                      for i, (rel, chunk)
                                      in enumerate(sorted_args)
                                      if chunk.is_wh_chunk()][0]

        pred_index = [i
                      for i, (rel, chunk)
                      in enumerate(sorted_args)
                      if rel == "PRED"][0]


        if ("subj" in wh_rel) \
           or (wh_index > pred_index):
            # No need to mess with the ordering
            return sorted_args

        # Find the correct location for the wh_chunk
        new_wh_loc = pred_index + 1
        for i in range(pred_index + 1,
                       len(sorted_args)):
            if not any([aux_rel in sorted_args[i][0]
                        for aux_rel in ["prep", "aux", "dobj"]]):
                break
            new_wh_loc += 1

        # Rebuild list
        return [(rel, chunk)
                for (rel, chunk) in sorted_args[: new_wh_loc]
                if chunk != wh_chunk] \
                   + [(wh_rel, wh_chunk)] \
                   + [elem
                      for elem
                      in sorted_args[new_wh_loc : ]]


    def get_sorted_elements(self):
        """
        Get a list of sorted chunks.
        """
        return self._fix_wh_location(sorted([("PRED", self.root)] + self.children,
                                           key = lambda (rel, chunk): chunk.chunk_index))


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
        words = text.split(' ')
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


def get_children(tok):
    """
    Get tok's children as list.
    """
    return list(tok.children)


def chunk_question(phrase):
    """
    Returns a ParsedQuestion instance, representing
    a given spacy parsed question phrase.
    """
    # Find head vp
    root = find_root(phrase)

    head_vp  = get_vp(root, phrase)
    ret = ParsedQuestion(head_vp)

    all_children = [(dep_, child)
                    for head in head_vp.toks
                    for (dep_, child) in collapsed_children(head)
                    if (child not in head_vp.toks)]

    if len([child
            for (dep, child) in all_children
            if (child.lemma_ == "do") \
            and (len(get_children(child)) == 1) \
            and ("obj" in get_children(child)[0].dep_)]):
        invert_subj = True
    else:
        invert_subj = False


    # Add a chunk for each child of the head chunk
    for (dep_, child) in all_children:
        # Special handling of "do" constructions
        if (child.lemma_ == "do"):
            if (len(get_children(child)) == 1) \
               and "obj" in get_children(child)[0].dep_:
                new_child = get_children(child)[0]
                ret.add_child_chunk(rel = "nsubj",
                                    child_chunk = Chunk(head = new_child,
                                                        toks = new_child.subtree,
                                                        side = Sides.LEFT if (new_child in root.lefts)\
                                                        else Sides.RIGHT))

        else:
            chunk = Chunk(head = child,
                          toks = child.subtree,
                          side = Sides.LEFT if (child in root.lefts)\
                          else Sides.RIGHT)

            if invert_subj and "subj" in dep_:
                # replace positions between subj and obj
                rel = "obj"

            else:
                rel = dep_

            ret.add_child_chunk(rel = rel,
                                child_chunk = chunk)  # Add the side of this chunk
    return ret


def collapse_child_rel(child):
    """
    Returns a collapsed version of the child.dep_
    Format:
    (collapsed_relation, child)
    """
    deps = list(child.children)

    # Check if need to collapse
    if is_prep(child) and\
       len(deps) == 1 and \
       is_pobj(deps[0]):
        return (("{}_{}".format(child.dep_,
                                child.lower_)), # Collapsed relation
                deps[0])

    else:
        return (child.dep_,
                child)


def collapsed_children(tok):
    """
    Returns a collapsed version of the toks' children.
    Format:
    List of (collapsed_relation, child)
    """
    return [collapse_child_rel(child)
            for child in tok.children]


def get_vp(head, question):
    """
    Return the VP rooted at a given head
    as a list of sorted spacy tokens
    """
    # Make sure we're dealing with verb phrase head
    toks = []
    new_toks = [head]

    while new_toks:
        buff = new_toks
        new_toks = []

        new_toks = [child
                    for cur_head in buff
                    for child in cur_head.children
                    if is_prt(child) \
                    and (not is_stranded_pp(child)) \
                    and (not ((child.lemma_ == "do") \
                              and not (is_neg_do(child, question))))]

        toks.extend(new_toks)

    # add "to" where needed
    if '{} {}'.format(question[-3], question[-2]) == "to do":
        toks.append(question[-3])

    return Chunk(head = head,
                 toks = sorted(toks,
                               key = lambda tok: tok.i),
                 side = Sides.MIDDLE)

def is_neg_do(child, question):
    """
    Returns True iff the given token
    is a "do" token which precedes a negation.
    """
    if (child.lemma_ != "do") \
       or (child.i == len(question) - 1):
        # This isn't do, or the last token
        return False

    return question[child.i + 1].dep_ == "neg"

def find_root(phrase):
    """
    Given a spacy parse, return its root
    https://spacy.io/docs/usage/dependency-parse
    """
    roots = [w for w in phrase if w.head is w]
    if (len(roots) != 1):
        logging.error("Encountered more than one root for '{}': {}".\
                      format(phrase, roots))
    return roots[0]


def is_aux(tok):
    """
    Is this token an auxiliary
    """
    return tok.dep_.startswith('aux')

def is_neg(tok):
    """
    Is this token a negation
    """
    return tok.dep_.startswith('neg')

def is_subj(tok):
    """
    Is this token a subject of its parent token.
    """
    return "subj" in tok.dep_

def is_prt(tok):
    """
    Is this token a particle.
    """
    return tok.dep_.startswith('prt') \
        or tok.dep_.startswith('xcomp') \
        or is_aux(tok) \
        or is_neg(tok) \
        or (is_prep(tok) and (len(list(tok.children)) == 0))


def is_prep(tok):
    """
    Is this token a preposition.
    """
    return tok.dep_.startswith('prep')

def is_pobj(tok):
    """
    Is this token an object of a preposition.
    """
    return tok.dep_.startswith('pobj')

def is_wh(tok):
    """
    Is this a wh pronoun.
    """
    return tok.tag_.startswith('W')

def is_verb(tok):
    """
    Is this token a verb
    """
    return tok.tag_.startswith('V')

def is_stranded_pp(child):
    """
    Whether this predicate has a stranded preposition
    E.g., "What is John associated with?"
    """
    return is_prep(child)  \
        and not any([is_pobj(tok)    # Doesn't have an object
                     for tok in child.children])

class SpacyRootError(Exception):
    """
    Exception when encountering a an unexpected
    number of roots in a spacy parse
    """
    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)

def left_tokens(token):
    """
    Get all tokens to the left of a given token
    """
    return [tok
            for left_tok in token.lefts
            for tok in left_tok.subtree]

def left_rels(token):
    """
    Get all dependency relations to the left of a given token
    """
    return [left_tok.dep_
            for left_tok in token.lefts]

def right_tokens(token):
    """
    Get all tokens to the right of a given token
    """
    return [tok
            for right_tok in token.rights
            for tok in right_tok.subtree]

def right_rels(token):
    """
    Get all dependency relations to the right of a given token
    """
    return [right_tok.dep_
            for right_tok in token.rights]

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
    return spacy_with_whitespace_tokenizer.parser(unicode(text,
                                                          encoding))

if __name__ == "__main__":
    """
    Sanity checks
    """
    doc1 = spacy_with_whitespace_tokenizer.parser(unicode("Who's the U.S. first president?"))
    doc2 = spacy_with_whitespace_tokenizer.parser(unicode("Who may call the new president ?"))
