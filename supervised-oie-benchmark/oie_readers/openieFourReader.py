from oie_readers.oieReader import OieReader
from oie_readers.extraction import Extraction

class OpenieFourReader(OieReader):
    
    def __init__(self):
        self.name = 'OpenIE-4'
    
    def read(self, fn):
        d = {}
        with open(fn) as fin:
            for line in fin:
                data = line.strip().split('\t')
                confidence = data[0]
                if not all(data[2:5]):
                    continue
                arg1, rel, arg2 = [s[s.index('(') + 1:s.index(',List(')] for s in data[2:5]]
                text = data[5]
                curExtraction = Extraction(pred = rel, sent = text, confidence = float(confidence))
                curExtraction.addArg(arg1)
                curExtraction.addArg(arg2)
                d[text] = d.get(text, []) + [curExtraction]
        self.oie = d
