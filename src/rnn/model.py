""" Usage:
    model --train=TRAIN_FN --test=TEST_FN
"""


import numpy as np
import pandas
from docopt import docopt
from keras.models import Sequential
from keras.layers import Dense, LSTM, Embedding, TimeDistributedDense, TimeDistributed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import logging
logging.basicConfig(level = logging.DEBUG)

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self,  sent_maxlen, batch_size, seed = 42, sep = '\t', vocab_size = 10000, hidden_units = 128, embedding_size = 300):
        """
        Initialize the model
        sent_maxlen - the maximum length in words of each sentence - will be used for padding / truncating
        batch_size - batch size for training
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        vocab_size - size of the language voacbaulary to be used
        hidden_units - number of hidden units per layer
        embedding_size - the word embedding dimension
        """
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        self.vocab_size = vocab_size
        np.random.seed(self.seed)
        self.encoder = LabelEncoder()
        self.hidden_units = hidden_units
        self.embedding_size = embedding_size

    def classes_(self):
        return self.encoder.classes_

    def set_estimator(self, model):
        """
        Set this rnn's model
        model - a function which returns a Keras model, this would be
        passed to the KerasClassifier function, which will call the fit function
        internally.
        """
        self.estimator = KerasClassifier(build_fn = model,
                                         nb_epoch = 200,
                                         batch_size = self.batch_size,
                                         verbose = 0)

    def kfold_evaluation(self, dataset_fn, n_splits = 10):
        """
        Perform k-fold evaluation given a dataset file (csv)
        """
        kfold = KFold(n_splits = n_splits, shuffle = True, random_state = self.seed)
        X, Y = self.load_dataset(dataset_fn)
        results = cross_val_score(self.estimator, X, Y, cv = kfold)
        logging.info("Results: {:.2f} ({:.2f})".format(results.mean()*100,
                                                       results.std() * 100))
        return results

    def train_and_test(self, train_fn, test_fn):
        """
        Train and then test on given files
        """
        self.train(train_fn)
        return self.test(test_fn)

    def train(self, train_fn):
        """
        Train this model on a given train dataset
        """
        X, Y = self.load_dataset(train_fn)
        self.set_oie_model() # Called here after labels have been identified in load dataset
        logging.debug("Training model on {}".format(train_fn))
        self.model.fit(X, Y, batch_size = self.batch_size)

    def test(self, test_fn):
        """
        Evaluate this model on a test file
        """
        X, Y = self.load_dataset(test_fn)
        self.predicted = np_utils.to_categorical(self.estimator.predict(X))
        acc = accuracy_score(Y, self.predicted) * 100
        logging.info("ACC: {:.2f}".format(acc))
        return acc

    def load_dataset(self, fn):
        """
        Load a supervised OIE dataset from file
        Assumes that the labels appear in the last column.
        """
        df = pandas.read_csv(fn, sep = self.sep, header = 0)

        # Encode one-hot representation of the labels
        self.encoder.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        return (np.array(self.encode_inputs(sents)),
                np.array(self.encode_outputs(sents)))

    def get_sents_from_df(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == i] for i in range(min(df.run_id), max(df.run_id))]

    def encode_inputs(self, sents):
        """
        Given a dataframe split to sentences, encode inputs for rnn classification.
        Should return a  sequence of sample of length maxlen.
        """
        # Encode inputs
        input_encodings = []
        for sent in sents:
            word_encodings = [one_hot(w, self.vocab_size, filters = "")[0] for w in sent.word.values]
            pred_word_encodings = [one_hot(w, self.vocab_size, filters = "")[0] for w in sent.pred.values]
            input_encodings.append([Sample(word, pred_word) for (word, pred_word) in
                                    zip(word_encodings, pred_word_encodings)])
        # Pad / truncate to desired maximum length
        ret = []
        for samples in pad_sequences(input_encodings,
                                     pad_func = lambda : Pad_sample(),
                                     maxlen = self.sent_maxlen):
            cur = []
            for sample in samples:
                cur.append(sample.encode())
            ret.append(cur)
        return ret


    def encode_outputs(self, sents):
        """
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        # Encode outputs
        for sent in sents:
            output_encodings.append(np_utils.to_categorical(self.encoder.transform(sent.label.values)))

        # Pad / truncate to maximum length
        return pad_sequences(output_encodings, lambda : np.array([0] * self.num_of_classes()), maxlen = self.sent_maxlen)


    def decode_label(self, encoded_label):
        """
        Decode a categorical representation of a label back to textual chunking label
        """
        return self.encoder.inverse_transform(encoded_label)

    @staticmethod
    def iris_model():
        """
        Return a baseline model for multi-class classification
        http://machinelearningmastery.com/multi-class-classification-tutorial-keras-deep-learning-library/
        """
        model = Sequential()
        model.add(Dense(4, input_dim = 4, init = "normal", activation = "relu"))
        model.add(Dense(3, init="normal", activation = "sigmoid"))
        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        return model

    def num_of_classes(self):
        """
        Return the number of ouput classes
        """
        return len(self.classes_())

    def set_oie_model(self):
        """
        Return a function which returns a Keras sequential model for predicting OIE
        this should be given as input to set_estimator
        """
        self.model = Sequential()
        self.model.add(TimeDistributed(Embedding(self.vocab_size, self.embedding_size, dropout=0.2), input_shape = (self.sent_maxlen, 1)))
        self.model.add(TimeDistributed(LSTM(self.hidden_units, input_shape = (self.sent_maxlen, self.embedding_size),  return_sequences = True)))
        self.model.add(TimeDistributed(LSTM(self.hidden_units, input_shape = (self.sent_maxlen, self.hidden_units), return_sequences = False)))
        self.model.add(TimeDistributed(Dense(output_dim = self.num_of_classes(), activation = 'sigmoid')))

        self.model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        logging.debug(self.model.summary())


class Sample:
    """
    Single sample representation.
    Containter which names spans in the input vector to simplify access
    """
    def __init__(self, word, pred_word):
        self.word = word
        self.pred_word = pred_word

    def encode(self):
        """
        Encode this sample as vector as input for rnn,
        Probably just concatenating members in the right order.
        """
        return [self.word]

class Pad_sample(Sample):
    """
    A dummy sample used for padding
    """
    def __init__(self):
        Sample.__init__(self, word = 0, pred_word = 0)

def pad_sequences(sequences, pad_func, maxlen = None):
    """
    Similar to keras.preprocessing.sequence.pad_sequence but using Sample as higher level
    abstraction.
    pad_func is a pad class generator.
    """
    ret = []
    if maxlen is None:
        maxlen = max(map(len, sequences))

    # Pad / truncate (done this way to deal with np.array)
    for sequence in sequences:
        cur_seq = list(sequence[:maxlen])
        cur_seq.extend([pad_func()] * (maxlen - len(sequence)))
        ret.append(cur_seq)
    return ret

if __name__ == "__main__":
    args = docopt(__doc__)
    train_fn = args["--train"]
    test_fn = args["--test"]
    rnn = RNN_model(sent_maxlen = 20, batch_size = 5)
    rnn.train(train_fn)

#    rnn.kfold_evaluation(train_fn)
