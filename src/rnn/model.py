""" Usage:
    model --train=TRAIN_FN --test=TEST_FN
"""


import numpy as np
import pandas
from docopt import docopt
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
import logging
logging.basicConfig(level = logging.DEBUG)

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self, model, seed = 42, sep = '\t'):
        """
        Initialize the model
        model - a function which returns a Keras model, this would be
                passed to the KerasClassifier function, which will call the fit function
                internally.
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        """
        self.seed = seed
        self.sep = sep
        np.random.seed(self.seed)
        self.estimator = KerasClassifier(build_fn = model,
                                         nb_epoch = 200,
                                         batch_size = 5,
                                         verbose = 0)
        self.encoder = LabelEncoder()

    def kfold_evaluation(self, dataset_fn, n_splits = 10):
        """
        Perform k-fold evaluation given a dataset file (csv)
        """
        kfold = KFold(n_splits = n_splits, shuffle = True, random_state = self.seed)
        X, Y = self.load_dataset(dataset_fn)
        results = cross_val_score(self.estimator, X, Y, cv = kfold)
        logging.info("Results: {:.2f} ({:.2f})".format(results.mean()*100,
                                                       results.std() * 100))

    def load_dataset(self, fn):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(fn, sep = self.sep, header = None)
        dataset = df.values
        num_of_feats = np.shape(dataset)[1]

        # Extract labels and samples
        self.samples = dataset[:, 0:(num_of_feats - 1)]
        self.labels  = dataset[:, (num_of_feats - 1)]

        # Encode one-hot representation of the labels
        self.encoder.fit(self.labels)

        return  self.encode(self.samples,
                            self.labels)

    def encode(self, inputs, outputs):
        """
        Prepare inputs and outputs for processing in the RNN
        """
        # TODO - handle encoding of inputs
        return inputs, np_utils.to_categorical(self.encoder.transform(outputs))

    def decode_label(self, encoded_label):
        """
        Decode a one-hot representation of a label back to chunking label
        """
        return self.encoder.inverse_transform(np.argmax(encoded_label))

    @staticmethod
    def baseline_model():
        """
        Return a baseline model for multi-class classification
        """
        model = Sequential()
        model.add(Dense(4, input_dim = 4, init = "normal", activation = "relu"))
        model.add(Dense(3, init="normal", activation = "sigmoid"))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model


if __name__ == "__main__":
    args = docopt(__doc__)
    train_fn = args["--train"]
    test_fn = args["--test"]
    rnn = RNN_model(model = RNN_model.baseline_model, sep = ',')
    rnn.kfold_evaluation(train_fn)
