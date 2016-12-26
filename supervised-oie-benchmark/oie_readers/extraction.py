from sklearn.preprocessing.data import binarize
from oie_readers.argument import Argument
from operator import itemgetter
from collections import defaultdict
import nltk
import itertools
import logging
import numpy as np

class Extraction:
    """
    Stores sentence, single predicate and corresponding arguments
    """
    def __init__(self, pred, sent, confidence):
        self.pred = pred
        self.sent = sent
        self.args = []
        self.confidence = confidence
        self.matched = []
        self.questions = {}
        self.indsForQuestions = defaultdict(lambda: set())

    def distArgFromPred(self, arg):
        assert(len(self.pred) == 2)
        dists = []
        for x in self.pred[1]:
            for y in arg.indices:
                dists.append(abs(x - y))

        return min(dists)

    def argsByDistFromPred(self, question):
        return sorted(self.questions[question], key = lambda arg: self.distArgFromPred(arg))

    def addArg(self, arg, question = None):
        logging.debug("Adding argument: {}".format(arg))
        self.args.append(arg)
        if question:
            self.questions[question] = self.questions.get(question,[]) + [Argument(arg)]

    def noPronounArgs(self):
        for (a, _) in self.args:
            logging.debug("POS tagging: {}".format(a))
            (_, pos) = nltk.pos_tag([a.lstrip().rstrip()])[0]
            if 'PRP' in pos:
                return False
        return True

    def isContiguous(self):
        return all([indices for (_, indices) in self.args])

    def toBinary(self):
        ''' Try to represent this extraction's arguments as binary
        If fails, this function will return an empty list.  '''

        ret = [self.elementToStr(self.pred)]

        if len(self.args) == 2:
            # we're in luck
            return ret + [self.elementToStr(arg) for arg in self.args]

        return []

        if not self.isContiguous():
            # give up on non contiguous arguments (as we need indexes)
            return []

        # otherwise, try to merge based on indices
        # TODO: you can explore other methods for doing this
        binarized = self.binarizeByIndex()

        if binarized:
            return ret + binarized

        return []


    def elementToStr(self, elem, print_indices = True):
        ''' formats an extraction element (pred or arg) as a raw string
        removes indices and trailing spaces '''
        if print_indices:
            return str(elem)
        if isinstance(elem, str):
            return elem
        if isinstance(elem, tuple):
            ret = elem[0].rstrip().lstrip()
        else:
            ret = ' '.join(elem.words)
        assert ret, "empty element? {0}".format(elem)
        return ret

    def binarizeByIndex(self):
        extraction = [self.pred] + self.args
        markPred = [(w, ind, i == 0) for i, (w, ind) in enumerate(extraction)]
        sortedExtraction = sorted(markPred, key = lambda (ws, indices, f) : indices[0])
        s =  ' '.join(['{1} {0} {1}'.format(self.elementToStr(elem), SEP) if elem[2] else self.elementToStr(elem) for elem in sortedExtraction])
        binArgs = [a for a in s.split(SEP) if a.rstrip().lstrip()]

        if len(binArgs) == 2:
            return binArgs

        # failure
        return []

    def bow(self):
        return ' '.join([self.elementToStr(elem) for elem in [self.pred] + self.args])

    def getSortedArgs(self):
        ls = []
        for q, args in self.questions.iteritems():
            if (len(args) != 1):
                logging.debug("Not one argument: {}".format(args))
                continue
            arg = args[0]
            indices = list(self.indsForQuestions[q].union(flatten(arg.indices)))
            if not indices:
                logging.warn("Empty indexes for arg {} -- backing to zero".format(arg))
                indices = [0]
            ls.append((arg, indices))
        return [a for a, _ in sorted(ls, key = lambda (_, indices): min(indices))]

    def clusterScore(self, cluster):
        """
        Calculate cluster density score as the mean distance of the maximum distance of each slot.
        Lower score represents a denser clusterOD
        """
        logging.debug("*-*-*- Cluster: {}".format(cluster))
        # Find global centroid
        arr = np.array([x for ls in cluster for x in ls])
        centroid = np.sum(arr)/arr.shape[0]
        logging.debug("Centroid: {}".format(centroid))
        # Calculate mean over all maxmimum points
        return np.average([max([abs(x - centroid) for x in ls]) for ls in cluster])

    def resolveAmbiguity(self):
        """
        Heursitic to map the elments (argument and predicates) of this extraction
        back to the indices of the sentence.
        """
        ## TODO: This removes arguments for which there was no consecutive span found
        ## Part of these are non-consecutive arguments, but other could be a bug in recognizing some punctuation marks
        elements = [self.pred] + [(s, indices) for (s, indices) in self.args if indices]
        logging.debug("Resolving ambiguity in: {}".format(elements))

        # Collect all possible combinations of arguments and predicate indices
        # (hopefully it's not too much)
        all_combinations = list(itertools.product(*map(itemgetter(1), elements)))
        logging.debug("Number of combinations: {}".format(len(all_combinations)))

        # Choose the ones with best clustering and unfold them
        resolved_elements = zip(map(itemgetter(0), elements),
                                min(all_combinations, key = lambda cluster: self.clusterScore(cluster)))
        logging.debug("Resolved elements = {}".format(resolved_elements))

        self.pred = resolved_elements[0]
        self.args = resolved_elements[1:]

    def conll(self):
        """
        Return a CoNLL string representation of this extraction
        """
        return '\n'.join(["\t".join((w, self.get_label(i)))
                          for (i, w) in enumerate(self.sent.split(" "))]) + '\n'

    def get_label(self, index):
        """
        Given an index of a word in the sentence -- returns the appropriate BIO conll label
        Assumes that ambiguation was already resolved.
        """
        # Get the element in which this index appears
        ent = [(elem_ind, elem) for (elem_ind, elem) in enumerate(map(itemgetter(1), [self.pred] + self.args)) if index in elem]
        if not ent:
            # index doesnt appear in any element
            return "O"
        assert len(ent) == 1, "Index {} appears in one than more element: {}".format(index, (ent, self.sent, self.pred, self.args))
        elem_ind, elem = ent[0]

        # Distinguish between predicate and arguments
        prefix = "P" if elem_ind == 0 else "A{}".format(elem_ind - 1)

        # Distinguish between Beginning and Inside labels
        suffix = "B" if index == elem[0] else "I"

        return "{}-{}".format(prefix, suffix)

    def __str__(self):
        return '{0}\t{1}'.format(self.elementToStr(self.pred, print_indices = True), '\t'.join([self.elementToStr(arg) for arg in self.args]))
#        return '{0}\t{1}'.format(self.elementToStr(self.pred, print_indices = True), '\t'.join([escape_special_chars(self.elementToStr(arg)) for arg in self.getSortedArgs()]))
#        return '{0}\t{1}'.format(self.elementToStr(self.pred, print_indices = True), '\t'.join([self.elementToStr(arg) for arg in self.getSortedArgs()]))

# Flatten a list of lists
flatten = lambda l: [item for sublist in l for item in sublist]




## CONSTANTS
SEP = ';;;'
