import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self, seed = 42):
        """
        Initialize the model
        seed - the random seed for reproduciblity
        """
        self.seed = seed
        numpy.random.seed(self.seed)
        self.model = Sequential()

        # for a multi-class classification problem
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def load_dataset(self, fn):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(fn, header = None)
