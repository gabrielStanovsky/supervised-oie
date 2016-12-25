from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class ClausieReader(OieReader):
    
    def __init__(self):
        self.name = 'ClausIE'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                if len(data) == 1:
                    text = data[0]
                elif len(data) == 5:
                    arg1, rel, arg2 = [s[1:-1] for s in data[1:4]]
                    confidence = data[4]
                    
                    curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence))
                    curExtraction.addArg(arg1)
                    curExtraction.addArg(arg2)
                    d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
        self.normalizeConfidence()
    
    
    def normalizeConfidence(self):
        ''' Normalize confidence to resemble probabilities '''        
        EPSILON = 1e-3
        
        confidences = [extraction.confidence for sent in self.oie for extraction in self.oie[sent]]
        maxConfidence = max(confidences)
        minConfidence = min(confidences)
        
        denom = maxConfidence - minConfidence + (2*EPSILON)
        
        for sent, extractions in self.oie.items():
            for extraction in extractions:
                extraction.confidence = ( (extraction.confidence - minConfidence) + EPSILON) / denom
