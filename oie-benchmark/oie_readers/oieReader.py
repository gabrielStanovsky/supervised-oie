class OieReader:
    
    def read(self, fn, includeNominal):
        ''' should set oie as a class member 
        as a dictionary of extractions by sentence'''
        raise Exception("Don't run me")
    
    def count(self):
        ''' number of extractions '''
        return sum([len(extractions) for _, extractions in self.oie.items()])