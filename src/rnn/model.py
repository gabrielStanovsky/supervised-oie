""" Usage:
    model --train=TRAIN_FN --test=TEST_FN [--glove=EMBEDDING]
"""
import numpy as np
import pandas
from docopt import docopt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributedDense, TimeDistributed
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from keras.preprocessing.text import one_hot
from keras.preprocessing import sequence
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from load_pretrained_word_embeddings import Glove
import logging
logging.basicConfig(level = logging.DEBUG)

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self,  model_fn, sent_maxlen, emb,
                 batch_size = 50, seed = 42, sep = '\t',
                 hidden_units = 128,trainable_emb = True,
                 emb_dropout = 0.2
    ):
        """
        Initialize the model
        model_fn - a model generating function, to be called when training with self as a single argument.
        sent_maxlen - the maximum length in words of each sentence - will be used for padding / truncating
        batch_size - batch size for training
        pre_trained_emb - an embedding class
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        hidden_units - number of hidden units per layer
        trainable_emb - controls if the loss should propagate to the word embeddings during training
        emb_dropout - the percentage of dropout during embedding
        """
        self.model_fn = model_fn
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        np.random.seed(self.seed)
        self.encoder = LabelEncoder()
        self.hidden_units = hidden_units
        self.emb = emb
        self.embedding_size = self.emb.dim
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout

    def classes_(self):
        return self.encoder.classes_

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
        self.model_fn(self)  # Set model params, called here after labels have been identified in load dataset
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
            word_encodings = [self.emb.get_word_index(w) for w in sent.word.values]
            pred_word_encodings = [self.emb.get_word_index(w) for w in sent.pred.values]
            input_encodings.append([Sample(word, pred_word) for (word, pred_word) in
                                    zip(word_encodings, pred_word_encodings)])

        # Pad / truncate to desired maximum length
        ret = []
        for samples in pad_sequences(input_encodings,
                                     pad_func = lambda : Pad_sample(),
                                     maxlen = self.sent_maxlen):
            ret.append([sample.encode() for sample in samples])
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
        return pad_sequences(output_encodings,
                             lambda : np.array([0] * self.num_of_classes()),
                             maxlen = self.sent_maxlen)


    def decode_label(self, encoded_label):
        """
        Decode a categorical representation of a label back to textual chunking label
        """
        return self.encoder.inverse_transform(encoded_label)

    def num_of_classes(self):
        """
        Return the number of ouput classes
        """
        return len(self.classes_())

    def set_vanilla_model(self):
        """
        Set a Keras sequential model for predicting OIE as a member of this class
        Can be passed as model_fn to the constructor
        https://keras.io/getting-started/functional-api-guide/
        """
        logging.debug("Setting vanilla model")
        # First layer
        ## Word embedding
        word_inputs = Input(shape = (self.sent_maxlen, 1), dtype="int32", name = "word_inputs")
        word_embeddings = TimeDistributed(self.emb.get_keras_embedding(dropout = self.emb_dropout,
                                                                       trainable = self.trainable_emb))\
                                                                       (word_inputs)
        # Deep layers
        deep = lambda inp:\
               TimeDistributed(LSTM(self.hidden_units,
                                            return_sequences = False)) \
                                            (TimeDistributed(LSTM(self.hidden_units,
                                                                  return_sequences = True)) (inp))

        predict = lambda inp:\
                  TimeDistributed(Dense(output_dim = self.num_of_classes(), activation = 'sigmoid'))(inp)

        output = predict(deep(word_embeddings))

        # Build model
        self.model = Model(input = [word_inputs], output = [output])

        # Loss
        self.model.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        self.model.summary()

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

    # Determine the maxlen -- Make sure it doesn't exceed the maximum observed length
    max_value = max(map(len, sequences))
    if maxlen is None:
        maxlen = max_value
        logging.debug("Padding to maximum observed length ({})".format(max_value))
    else:
        maxlen = min(max_value, maxlen)
        logging.debug("Padding / truncating to {} words (max observed was {})".format(maxlen, max_value))
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
    if "--glove" in args:
        emb = Glove(args["--glove"])
        rnn = RNN_model(model_fn = RNN_model.set_vanilla_model, sent_maxlen = None, emb = emb)
        rnn.train(train_fn)



"""
Things to do:

 1. Set pretrained embeddings
 2. Add features (merge rnns?)
 3. Test performance.

"""
