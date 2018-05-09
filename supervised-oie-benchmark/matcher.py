import string
from nltk.translate.bleu_score import sentence_bleu
from nltk.corpus import stopwords

class Matcher:
    @staticmethod
    def bowMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        A binary function testing for exact lexical match (ignoring ordering) between reference
        and predicted extraction
        """
        s1 = ref.bow()
        s2 = ex.bow()
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return sorted(s1Words) == sorted(s2Words)

    @staticmethod
    def predMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the predicate
        """
        s1 = ref.elementToStr(ref.pred)
        s2 = ex.elementToStr(ex.pred)
        if ignoreCase:
            s1 = s1.lower()
            s2 = s2.lower()

        s1Words = s1.split(' ')
        s2Words = s2.split(' ')

        if ignoreStopwords:
            s1Words = Matcher.removeStopwords(s1Words)
            s2Words = Matcher.removeStopwords(s2Words)

        return s1Words  == s2Words


    @staticmethod
    def argMatch(ref, ex, ignoreStopwords, ignoreCase):
        """
        Return whehter gold and predicted extractions agree on the arguments
        """
        sRef = ' '.join([ref.elementToStr(elem) for elem in ref.args])
        sEx = ' '.join([ex.elementToStr(elem) for elem in ex.args])

        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)


        return coverage > Matcher.LEXICAL_THRESHOLD

    @staticmethod
    def bleuMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow()
        sEx = ex.bow()
        bleu = sentence_bleu(references = [sRef.split(' ')], hypothesis = sEx.split(' '))
        return bleu > Matcher.BLEU_THRESHOLD

    @staticmethod
    def lexicalMatch(ref, ex, ignoreStopwords, ignoreCase):
        sRef = ref.bow().split(' ')
        sEx = ex.bow().split(' ')
        count = 0

        for w1 in sRef:
            for w2 in sEx:
                if w1 == w2:
                    count += 1

        # We check how well does the extraction lexically cover the reference
        # Note: this is somewhat lenient as it doesn't penalize the extraction for
        #       being too long
        coverage = float(count) / len(sRef)


        return coverage > Matcher.LEXICAL_THRESHOLD

    @staticmethod
    def removeStopwords(ls):
        return [w for w in ls if w.lower() not in Matcher.stopwords]

    # CONSTANTS
    BLEU_THRESHOLD = 0.4
    LEXICAL_THRESHOLD = 0.5 # Note: changing this value didn't change the ordering of the tested systems
    stopwords = stopwords.words('english') + list(string.punctuation)





