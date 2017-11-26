""" Usage:
    qa_to_oie --in=INPUT_FILE --out=OUTPUT_FILE --conll=CONLL_FILE [--dist=DIST_FILE] [--oieinput=OIE_INPUT] [-v]
"""

from docopt import docopt
import re
import itertools
from oie_readers.extraction import Extraction, escape_special_chars, normalize_element
from collections  import defaultdict
import logging
import operator
from fuzzywuzzy import process
from fuzzywuzzy.utils import full_process
import itertools
from fuzzywuzzy.string_processing import StringProcessor
from fuzzywuzzy.utils import asciidammit
from operator import itemgetter
import nltk
import json
import pdb

from oie_readers.extraction import QUESTION_TRG_INDEX
from oie_readers.extraction import QUESTION_PP_INDEX
from oie_readers.extraction import QUESTION_OBJ2_INDEX



## CONSTANTS

QUESTION_TRG_INDEX =  3 # index of the predicate within the question
QUESTION_MODALITY_INDEX = 1 # index of the modality within the question
PASS_ALL = lambda x: x
MASK_ALL = lambda x: "_"
get_default_mask = lambda : [PASS_ALL] * 8

# QA-SRL vocabulary for "AUX" placement, which modifies the predicates
QA_SRL_AUX_MODIFIERS = [
 #   "are",
    "are n't",
    "can",
    "ca n't",
    "could",
    "could n't",
#    "did",
    "did n't",
#    "do",
#    "does",
    "does n't",
    "do n't",
    "had",
    "had n't",
#    "has",
    "has n't",
#    "have",
    "have n't",
#    "is",
    "is n't",
    "may",
    "may not",
    "might",
    "might not",
    "must",
    "must n't",
    "should",
    "should n't",
#    "was",
    "was n't",
#    "were",
    "were n't",
    "will",
    "wo n't",
    "would",
    "would n't",
]



class Qa2OIE:
    # Static variables
    extractions_counter = 0

    def __init__(self, qaFile, dist_file = ""):
        """
        Loads qa file and converts it into  open IE
        If a distribtion file is given, it is used to determine the hopefully correct
        order of arguments. Otherwise, these are oredered accroding to their linearization
        """
        # This next lines ensures that the json is loaded with numerical
        # indexes for loc
        self.question_dist = dict([(q, dict([(int(loc), cnt)
                                             for (loc, cnt)
                                             in dist.iteritems()]))
                                   for (q, dist)
                                   in json.load(open(dist_file)).iteritems()]) \
                                       if dist_file\
                                          else {}

        self.dic = self.loadFile(self.getExtractions(qaFile))

    def loadFile(self, lines):
        sent = ''
        d = {}

        indsForQuestions = defaultdict(lambda: set())

        for line in lines.split('\n'):
            line = line.strip()
            if not line:
                continue
            data = line.split('\t')
            if len(data) == 1:
                if sent:
                    for ex in d[sent]:
                        ex.indsForQuestions = dict(indsForQuestions)
                sent = line
                d[sent] = []
                indsForQuestions = defaultdict(lambda: set())

            else:
                pred = self.preproc(data[0])
                pred_index = map(int,
                                 eval(data[1]))
                cur = Extraction((pred,
                                  [pred_index]),
                                 sent,
                                 confidence = 1.0)

                for q, a in zip(data[2::2], data[3::2]):
                    preproc_arg = self.preproc(a)
                    if not preproc_arg:
                        logging.warn("Argument reduced to None: {}".format(a))
                    indices = fuzzy_match_phrase(preproc_arg.split(" "),
                                                 sent.split(" "))
                    cur.addArg((preproc_arg, indices), q)
                    indsForQuestions[q] = indsForQuestions[q].union(flatten(indices))


                if sent:
                    if cur.noPronounArgs():
                        cur.resolveAmbiguity()
                        d[sent].append(cur)

        return d

    def preproc(self, s):
        """
        Returns a unified preproc of a string:
          - Removes duplicates spaces, to allow for space delimited words.
        """
        return " ".join([w for w in s.split(" ") if w])

    def getExtractions(self, qa_srl_path, mask = get_default_mask()):
        """
        Parse a QA-SRL file (with raw sentences) at qa_srl_path.
        Returns output which can in turn serve as input for load_file.
        """
        lc = 0
        sentQAs = []
        curAnswers = []
        curSent = ""
        ret = ''

        for line in open(qa_srl_path, 'r'):
            if line.startswith('#'):
                continue
            line = line.strip()
            info = line.strip().split("\t")
            if lc == 0:
                # Read sentence ID.
                sent_id = int(info[0].split("_")[1])
                ptb_id = []
                lc += 1
            elif lc == 1:
                if curSent:
                    ret += self.printSent(curSent, sentQAs)
                # Write sentence.
                curSent = line
                lc += 1
                sentQAs = []
            elif lc == 2:
                if curAnswers:
                    sentQAs.append(((surfacePred,
#                                     predIndex),
                                     augmented_pred_indices),
                                    curAnswers))
                curAnswers = []
                # Update line counter.
                if line.strip() == "":
                    lc = 0 # new line for new sent
                else:
                    # reading predicate and qa pairs
                    predIndex, basePred, count = info
                    surfacePred = basePred
                    lc += int(count)
            elif lc > 2:
                question = encodeQuestion("\t".join(info[:-1]), mask)
                curSurfacePred = augment_pred_with_question(basePred, question)
                if len(curSurfacePred) > len(surfacePred):
                    surfacePred = curSurfacePred
                answers = self.consolidate_answers(info[-1].split("###"))
                curAnswers.append(zip([question]*len(answers), answers))

                lc -= 1
                if (lc == 2):
                    # Reached the end of this predicate's questions
                    # TODO: make sure that base pred is in the indices returned
                    #       by fuzzy matching
                    augmented_pred_indices = fuzzy_match_phrase(surfacePred.split(" "),
                                                                curSent.split(" "))
#                    pdb.set_trace()
                    if not augmented_pred_indices:
                        augmented_pred_indices = [predIndex]

                    else:
                        augmented_pred_indices = augmented_pred_indices[0]
#                    pdb.set_trace()
                    sentQAs.append(((surfacePred,
#                                     predIndex),
                                     augmented_pred_indices),
                                    curAnswers))
                    curAnswers = []
        # Flush
        if sentQAs:
            ret += self.printSent(curSent, sentQAs)

        return ret

    def printSent(self, sent, sentQAs):
        ret =  sent + "\n"
        for (pred, pred_index), predQAs in sentQAs:
            for element in itertools.product(*predQAs):
                self.encodeExtraction(element)
                ret += "\t".join([pred, str(pred_index)] +
                                 ["\t".join(x) for x in element]) + "\n"
        ret += "\n"
        return ret

    def encodeExtraction(self, element):
        questions = map(operator.itemgetter(0),element)
        extractionSet = set(questions)
        encoding = repr(extractionSet)
        (count, _, extractions) = extractionsDic.get(encoding, (0, extractionSet, []))
        extractions.append(Qa2OIE.extractions_counter)
        Qa2OIE.extractions_counter += 1
        extractionsDic[encoding] = (count+1, extractionSet, extractions)


    def consolidate_answers(self, answers):
        """
        For a given list of answers, returns only minimal answers - e.g., ones which do not
        contain any other answer in the set.
        This deals with certain QA-SRL anntoations which include a longer span than that is needed.
        """
        ret = []
        for i, first_answer in enumerate(answers):
            includeFlag = True
            for j, second_answer in enumerate(answers):
                if (i != j) and (is_str_subset(second_answer, first_answer)) :
                    includeFlag = False
                    continue
            if includeFlag:
                ret.append(first_answer)
        return ret

    def createOIEInput(self, fn):
        with open(fn, 'a') as fout:
            for sent in self.dic:
                fout.write(sent + '\n')

    def writeOIE(self, fn):
        with open(fn, 'w') as fout:
            for sent, extractions in self.dic.iteritems():
                for ex in extractions:
                    fout.write('{}\t{}\n'.format(escape_special_chars(sent),
                                                 ex.__str__()))
    def writeConllFile(self, fn):
        """
        Write a conll representation of all of the extractions to file
        """
        running_index = 0 # Running index enumerates the predicates in the dataset
        # Add a header file identifying each column
        header = '\t'.join(["word_id",
                            "word",
                            "pred",
                            "pred_id",
                            "sent_id",
                            "run_id",
                            "label"])

        with open(fn, 'w') as fout:
            fout.write(header + '\n')
            for sent_index, extractions in enumerate(self.dic.itervalues()):
                for ex in extractions:
                    fout.write(ex.conll(external_feats = [sent_index, running_index]) + '\n')
                    running_index += 1

# MORE HELPER

def augment_pred_with_question(pred, question):
    """
    Decide what elements from the question to incorporate in the given
    corresponding predicate
    """
    # Parse question
    wh, aux, sbj, trg, obj1, pp, obj2 = map(normalize_element,
                                            question.split(' ')[:-1]) # Last split is the question mark

    # Add auxiliary to the predicate
    if aux in QA_SRL_AUX_MODIFIERS:
        return " ".join([aux, pred])

    # Non modified predicates
    return pred


def is_str_subset(s1, s2):
    """ returns true iff the words in string s1 are contained in string s2 in the same order by which they appear in s2 """
    all_indices = [find_all_indices(s2.split(" "), x) for x in s1.split()]
    if not all(all_indices):
        return False
    for combination in itertools.product(*all_indices):
        if strictly_increasing(combination):
            return True
    return False

def find_all_indices(ls, elem):
    return  [i for i,x in enumerate(ls) if x == elem]

def strictly_increasing(L):
    return all(x<y for x, y in zip(L, L[1:]))

def is_consecutive(ls):
    """
    Returns true iff ls represents a list of consecutive numbers.
    """
    return all((y - x == 1) for x, y in zip(ls, ls[1:]))

questionsDic = {}
extractionsDic = {}

def encodeQuestion(question, mask):
    info = [mask[i](x).replace(" ","_") for i,x in enumerate(question.split("\t"))]
    encoding = "\t".join(info)
    # get the encoding of a question, and the count of times it appeared
    (val, count) = questionsDic.get(encoding, (len(questionsDic), 0))
    questionsDic[encoding] = (val, count+1)
    ret = " ".join(info)
    return ret

def all_index(s, ss, matchCase = True, ignoreSpaces = True):
    ''' Find all occurrences of substring ss in s '''
    if not matchCase:
        s = s.lower()
        ss = ss.lower()

    if ignoreSpaces:
        s = s.replace(' ', '')
        ss = ss.replace(' ','')

    return [m.start() for m in re.finditer(re.escape(ss), s)]


def fuzzy_match_phrase(phrase, sentence):
    """
    Fuzzy find the indexes of all word in phrase against a given sentence (both are lists of words),
    returns a list of indexes in the length of phrase which match the best return from fuzzy.
    """
    logging.debug("Fuzzy searching \"{}\" in \"{}\"".format(" ".join(phrase), " ".join(sentence)))
    limit = min((len(phrase) / 2) + 1, 3)
    possible_indices = [fuzzy_match_word(w,
                                         sentence,
                                         limit) \
                        + (fuzzy_match_word("not",
                                           sentence,
                                           limit) \
                           if w == "n't" \
                           else [])
                        for w in phrase]
    indices = find_consecutive_combinations(*possible_indices)
    if not indices:
        logging.warn("\t".join(map(str, ["*** {}".format(len(indices)),
                                         " ".join(phrase),
                                         " ".join(sentence),
                                         possible_indices,
                                         indices])))
    return indices


def find_consecutive_combinations(*lists):
    """
    Given a list of lists of integers, find only the consecutive options from the Cartesian product.
    """
    ret = []
    desired_length = len(lists) # this is the length of a valid walk
    logging.debug("desired length: {}".format(desired_length))
    for first_item in lists[0]:
        logging.debug("starting with {}".format(first_item))
        cur_walk = [first_item]
        cur_item = first_item
        for ls_ind, ls in enumerate(lists[1:]):
            logging.debug("ls = {}".format(ls))
            for cur_candidate in ls:
                if cur_candidate - cur_item == 1:
                    logging.debug("Found match: {}".format(cur_candidate))
                    # This is a valid option from this list,
                    # add it and break out of this list
                    cur_walk.append(cur_candidate)
                    cur_item = cur_candidate
                    break
            if len(cur_walk) != ls_ind + 2:
                # Didn't find a valid candidate -
                # break out of this first item
                break

        if len(cur_walk) == desired_length:
            ret.append(cur_walk)
    return ret


def fuzzy_match_word(word, words, limit):
    """
    Fuzzy find the indexes of word in words, returns a list of indexes which match the
    best return from fuzzy.
    limit controls the number of choices to allow.
    """
    # Try finding exact matches
    exact_matches = set([i for (i, w) in enumerate(words) if w == word])
    if exact_matches:
        logging.debug("Found exact match for {}".format(word))

    # Else, return fuzzy matching
    logging.debug("No exact match for: {}".format(word))
    # Allow some variance which extractOne misses
    # For example: "Armstrong World Industries Inc" in "Armstrong World Industries Inc. agreed in principle to sell its carpet operations to Shaw Industries Inc ."
    best_matches  = [w for (w, s) in process.extract(word, words, processor = semi_process, limit = limit) if (s > 70)]
    logging.debug("Best matches = {}".format(best_matches))
    return list(exact_matches.union([i for (i, w) in enumerate(words) if w in best_matches]))


# Flatten a list of lists
flatten = lambda l: [item for sublist in l for item in sublist]


def semi_process(s, force_ascii=False):
    """
    Variation on Fuzzywuzzy's full_process:
    Process string by
    XX removing all but letters and numbers --> These are kept to keep consecutive spans
    -- trim whitespace
    XX force to lower case --> These are kept since annotators marked verbatim spans, so case is a good signal
    if force_ascii == True, force convert to ascii
    """

    if s is None:
        return ""

    if force_ascii:
        s = asciidammit(s)
    # Remove leading and trailing whitespaces.
    string_out = StringProcessor.strip(s)
    return string_out



## MAIN
if __name__ == '__main__':
    args = docopt(__doc__)
    if args['-v']:
        logging.basicConfig(level = logging.DEBUG)
    else:
        logging.basicConfig(level = logging.INFO)
    logging.debug(args)
    inp = args['--in']
    out = args['--out']
    dist_file = args['--dist'] if args['--dist']\
           else ''
    q = Qa2OIE(args['--in'], dist_file = dist_file)
    q.writeOIE(args['--out'])
    q.writeConllFile(args['--conll'])
    if args['--oieinput']:
        q.createOIEInput(args['--oieinput'])
