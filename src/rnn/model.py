""" Usage:
    model [--train=TRAIN_FN] [--dev=DEV_FN] --test=TEST_FN [--epochs=EPOCHS] (--glove=EMBEDDING | --pretrained=MODEL_DIR)
"""
import numpy as np
import pandas
import nltk
from docopt import docopt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributedDense, TimeDistributed, merge, Bidirectional, Dropout
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
from operator import itemgetter
from keras.callbacks import LambdaCallback, ModelCheckpoint
from sklearn import metrics

import os
import json
from keras.models import model_from_json
import logging
logging.basicConfig(level = logging.DEBUG)

class RNN_model:
    """
    Represents an RNN model for supervised OIE
    """
    def __init__(self,  model_fn, sent_maxlen = None, emb_filename = None,
                 batch_size = 5, seed = 42, sep = '\t',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 classes = None,
    ):
        """
        Initialize the model
        model_fn - a model generating function, to be called when
                   training with self as a single argument.
        sent_maxlen - the maximum length in words of each sentence -
                      will be used for padding / truncating
        emb_filename - the filename from which to load the embedding
                       (Currenly only Glove. Idea: parse by filename)
        batch_size - batch size for training
        pre_trained_emb - an embedding class
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        hidden_units - number of hidden units per layer
        trainable_emb - controls if the loss should propagate to the word embeddings during training
        emb_dropout - the percentage of dropout during embedding
        num_of_latent_layers - how many LSTMs to stack
        epochs - the number of epochs to train the model
        pred_dropout - the proportion to dropout before prediction
        model_dir - the path in which to save model
        classes - the classes to be encoded (list of strings)
        """
        self.model_fn = lambda : model_fn(self)
        self.model_dir = model_dir
        self.sent_maxlen = sent_maxlen
        self.batch_size = batch_size
        self.seed = seed
        self.sep = sep
        self.encoder = LabelEncoder()
        self.hidden_units = hidden_units
        self.emb_filename = emb_filename
        self.emb = Glove(emb_filename)
        self.embedding_size = self.emb.dim
        self.trainable_emb = trainable_emb
        self.emb_dropout = emb_dropout
        self.num_of_latent_layers = num_of_latent_layers
        self.epochs = epochs
        self.pred_dropout = pred_dropout
        self.classes = classes

        np.random.seed(self.seed)

    def get_callbacks(self, X):
        """
        Sets these callbacks as a class member.
        X is the encoded dataset used to print a sample of the output.
        Callbacks created:
        1. Sample output each epoch
        2. Save best performing model each epoch
        """

        sample_output_callback = LambdaCallback(on_epoch_end = lambda epoch, logs:\
                                                pprint(self.sample_labels(self.model.predict(X))))
        checkpoint = ModelCheckpoint(os.path.join(self.model_dir,
                                                  "{epoch:02d}-{val_categorical_accuracy:.2f}.hdf5"),
                                     verbose = 1,
                                     save_best_only = True)   # TODO: is there a way to save by best val_acc?

        return [#sample_output_callback,
                checkpoint]

    def plot(self, fn, train_fn):
        """
        Plot this model to an image file
        Train file is needed as it influences the dimentions of the RNN
        """
        from keras.utils.visualize_util import plot
        X, Y = self.load_dataset(train_fn)
        self.model_fn()
        plot(self.model, to_file = fn)

    def classes_(self):
        """
        Return the classes which are classified by this model
        """
        try:
            return self.encoder.classes_
        except:
            return self.classes

    def train_and_test(self, train_fn, test_fn):
        """
        Train and then test on given files
        """
        logging.info("Training..")
        self.train(train_fn)
        logging.info("Testing..")
        return self.test(test_fn)
        logging.info("Done!")

    def train(self, train_fn, dev_fn):
        """
        Train this model on a given train dataset
        Dev test is used for model checkpointing
        """
        X_train, Y_train = self.load_dataset(train_fn)
        X_dev, Y_dev = self.load_dataset(dev_fn)
        logging.debug("Classes: {}".format((self.num_of_classes(), self.classes_())))
        # Set model params, called here after labels have been identified in load dataset
        self.model_fn()

        # Create a callback to print a sample after each epoch
        logging.debug("Training model on {}".format(train_fn))
        self.model.fit(X_train, Y_train,
                       batch_size = self.batch_size,
                       nb_epoch = self.epochs,
                       validation_data = (X_dev, Y_dev),
                       callbacks = self.get_callbacks(X_train))

    @staticmethod
    def consolidate_labels(labels):
        """
        Return a consolidated list of labels, e.g., O-A1 -> O, A1-I -> A
        """
        return map(RNN_model.consolidate_label , labels)

    @staticmethod
    def consolidate_label(label):
        """
        Return a consolidated label, e.g., O-A1 -> O, A1-I -> A
        """
        return label.split("-")[0] if label.startswith("O") else label


    def predict_sentence(self, sent):
        """
        Return a predicted label for each word in an arbitrary length sentence
        sent - a list of string tokens
        """
        ret = []

        # Extract predicates by looking at verbal POS
        preds = [pred_word for (pred_word, pos) in nltk.pos_tag(sent)
                 if pos.startswith("V")]

        # Run RNN for each predicate on this sentence
        for pred in preds:
            cur_sample = self.create_sample(sent, pred)
            X = self.encode_inputs([cur_sample])
            ret.append([(self.consolidate_label(label), prob) for [(label, prob)] in
                        self.transform_output_probs(self.model.predict(X), get_prob = True)[0]])
        return ret

        # # Create instances for all verbs as possible predicates
        # self.X = self.encode_inputs([self.create_sample(sent, pred_word)
        #                         for pred_word in ])

        # return [(self.consolidate_label(label), prob)
        #         for [(label, prob)] in self.transform_output_probs(self.model.predict(self.X), get_prob = True)[0]]



        #return [RNN_model.consolidate_labels(np.array(self.transform_output_probs(self.model.predict(x))).flatten())
        #        for x in X]

        # [  self.model.predict(x)]
        # X = self.encode_inputs([sent])
        # # Get most probable predictions and flatten
        # y = RNN_model.consolidate_labels(np.array(rnn.transform_output_probs(y)).flatten())
        # return y

    def create_sample(self, sent, pred_word):
        """
        Return a dataframe which could be given to encode_inputs
        """
        logging.debug("Creating sample with pred: {}".format(pred_word))
        return pandas.DataFrame({"word": sent,
                                 "pred": [pred_word] * len(sent)})

    def test(self, test_fn, eval_metrics):
        """
        Evaluate this model on a test file
        eval metrics is a list composed of:
        (name, f: (y_true, y_pred) -> float (some performance metric))
        Prints and returns the metrics name and numbers
        """
        # Load gold and predict
        X, Y = self.load_dataset(test_fn)
        y = self.model.predict(X)

        # Get most probable predictions and flatten
        Y = RNN_model.consolidate_labels(np.array(self.transform_output_probs(Y)).flatten())
        y = RNN_model.consolidate_labels(np.array(self.transform_output_probs(y)).flatten())

        # Run evaluation metrics and report
        ret = []
        for (metric_name, metric_func) in eval_metrics:
            ret.append((metric_name, metric_func(Y, y)))
            logging.debug("calculating {}".format(ret[-1]))

        for (metric_name, metric_val) in ret:
            logging.info("{}: {:.4f}".format(metric_name,
                                             metric_val))
        return Y, y, ret

    def load_dataset(self, fn):
        """
        Load a supervised OIE dataset from file
        """
        df = pandas.read_csv(fn, sep = self.sep, header = 0)

        # Encode one-hot representation of the labels
        if self.classes_() is None:
            self.encoder.fit(df.label.values)

        # Split according to sentences and encode
        sents = self.get_sents_from_df(df)
        return (self.encode_inputs(sents),
                self.encode_outputs(sents))

    def get_sents_from_df(self, df):
        """
        Split a data frame by rows accroding to the sentences
        """
        return [df[df.run_id == i] for i in range(min(df.run_id), max(df.run_id))]

    def encode_inputs(self, sents):
        """
        Given a dataframe split to sentences, encode inputs for rnn classification.
        Should return a dictionary of sequences of sample of length maxlen.
        """
        # Encode inputs
        word_inputs = []
        pred_inputs = []
        for sent in sents:
            word_encodings = [self.emb.get_word_index(w) for w in sent.word.values]
            pred_word_encodings = [self.emb.get_word_index(w) for w in sent.pred.values]
            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])

        # Pad / truncate to desired maximum length
        ret = {"word_inputs" : [],
               "predicate_inputs": []}

        for name, sequence in zip(["word_inputs", "predicate_inputs"],
                                  [word_inputs, pred_inputs]):
            for samples in pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = self.sent_maxlen):
                ret[name].append([sample.encode() for sample in samples])

        return {k: np.array(v) for k, v in ret.iteritems()}


    def encode_outputs(self, sents):
        """
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        # Encode outputs
        for sent in sents:
            output_encodings.append(np_utils.to_categorical(self.transform_labels(sent.label.values)))

        # Pad / truncate to maximum length
        return np.array(pad_sequences(output_encodings,
                                      lambda : np.array([0] * self.num_of_classes()),
                                      maxlen = self.sent_maxlen))

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        # return self.encoder.transform(labels)
        classes  = list(self.classes_())
        return [classes.index(label) for label in labels]

    def transform_output_probs(self, y, get_prob = False):
        """
        Given a list of probabilities over labels, get the textual representation of the
        most probable assignment
        """
        return self.sample_labels(y,
                                  num_of_sents = len(y), # all sentences
                                  num_of_samples = max(map(len, y)), # all words
                                  num_of_classes = 1, # Only top probability
                                  start_index = 0, # all sentences
                                  get_prob = get_prob, # Indicate whether to get only labels
        )

    def inverse_transform_labels(self, indices):
        """
        Encode a list of textual labels
        """
        classes = self.classes_()
        return [classes[ind] for ind in indices]

    def num_of_classes(self):
        """
        Return the number of ouput classes
        """
        return len(self.classes_())

    # Functional Keras -- all of the following are currying functions expecting models as input
    # https://keras.io/getting-started/functional-api-guide/

    def embed(self):
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_keras_embedding(dropout = self.emb_dropout,
                                            trainable = self.trainable_emb,
                                            input_length = self.sent_maxlen)

    def predict_classes(self):
        """
        Predict to the number of classes
        Named arguments are passed to the keras function
        """
        return lambda x: self.stack(x,
                                    [lambda : TimeDistributed(Dense(output_dim = self.num_of_classes(),
                                                                    activation = "softmax"))] +
                                    [lambda : TimeDistributed(Dense(self.hidden_units,
                                                                    activation='relu'))] * 3)
    def stack_latent_layers(self, n):
        """
        Stack n bidi LSTMs
        """
        return lambda x: self.stack(x, [lambda : Bidirectional(LSTM(self.hidden_units,
                                                                    return_sequences = True))] * n )

    def stack(self, x, layers):
        """
        Stack layers (FIFO) by applying recursively on the output,
        until returing the input as the base case for the recursion
        """
        if not layers:
            return x # Base case of the recursion is the just returning the input
        else:
            return layers[0]()(self.stack(x, layers[1:]))

    def set_model_from_file(self):
        """
        Receives an instance of RNN and returns a model from the self.model_dir
        path which should contain a file named: model.json,
        and a single file with the hdf5 extension.
        Note: Use this function for a pretrained model, running model training
        on the loaded model will override the files in the model_dir
        """
        from glob import glob

        weights_fn = glob(os.path.join(self.model_dir, "*.hdf5"))
        assert len(weights_fn) == 1, "More/Less than one weights file in {}: {}".format(self.model_dir,
                                                                                        weights_fn)
        weights_fn = weights_fn[0]
        logging.debug("Weights file: {}".format(weights_fn))
        self.model = model_from_json(open(os.path.join(self.model_dir,
                                                       "./model.json")).read())
        self.model.load_weights(weights_fn)
        self.model.compile(optimizer="adam",
                           loss='categorical_crossentropy',
                           metrics = ["accuracy"])

    def set_vanilla_model(self):
        """
        Set a Keras model for predicting OIE as a member of this class
        Can be passed as model_fn to the constructor
        """
        logging.debug("Setting vanilla model")
        # Build model

        ## Embedding Layer
        embedding_layer = self.embed()

        ## Deep layers
        latent_layers = self.stack_latent_layers(self.num_of_latent_layers)

        # ## Dropout
        dropout = Dropout(self.pred_dropout)

        ## Prediction
        predict_layer = self.predict_classes()

        ## Prepare input features, and indicate how to embed them
        inputs_and_embeddings = [(Input(shape = (self.sent_maxlen,),
                                       dtype="int32",
                                       name = "word_inputs"),
                                  embedding_layer),
                                 (Input(shape = (self.sent_maxlen,),
                                       dtype="int32",
                                        name = "predicate_inputs"),
                                  embedding_layer)]

        ## Concat all inputs and run on deep network
        output = predict_layer(dropout(latent_layers(merge([embed(inp)
                                                            for inp, embed in inputs_and_embeddings],
                                                           mode = "concat",
                                                           concat_axis = -1))))

        # Build model
        self.model = Model(input = map(itemgetter(0), inputs_and_embeddings),
                           output = [output])

        # Loss
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "model.json"))

    def to_json(self):
        """
        Encode a json of the parameters needed to reload this model
        """
        return {
            "sent_maxlen": self.sent_maxlen,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "sep": self.sep,
            "classes": list(self.classes_()),
            "hidden_units": self.hidden_units,
            "trainable_emb": self.trainable_emb,
            "emb_dropout": self.emb_dropout,
            "num_of_latent_layers": self.num_of_latent_layers,
            "epochs": self.epochs,
            "pred_dropout": self.pred_dropout,
            "emb_filename": self.emb_filename,
        }

    def save_model_to_file(self, fn):
        """
        Saves this model to file, also encodes class inits in the model's json
        """
        js = json.loads(self.model.to_json())

        # Add this model's params
        js["rnn"] = self.to_json()
        with open(fn, 'w') as fout:
            json.dump(js, fout)


    def sample_labels(self, y, num_of_sents = 5, num_of_samples = 10,
                      num_of_classes = 3, start_index = 5, get_prob = True):
        """
        Get a sense of how labels in y look like
        """
        classes = self.classes_()
        ret = []
        for sent in y[:num_of_sents]:
            cur = []
            for word in sent[start_index: start_index + num_of_samples]:
                sorted_prob = am(word)
                cur.append([(classes[ind], word[ind]) if get_prob else classes[ind]
                            for ind in sorted_prob[:num_of_classes]])
            ret.append(cur)
        return ret

class Sentence:
    """
    Prepare sentence sample for encoding
    """
    def __init__(words, pred_index):
        """
        words - list of strings representing the words in the sentence.
        pred_index - int representing the index of the current predicate for which to predict OIE extractions
        """

class Sample:
    """
    Single sample representation.
    Containter which names spans in the input vector to simplify access
    """
    def __init__(self, word):
        self.word = word

    def encode(self):
        """
        Encode this sample as vector as input for rnn,
        Probably just concatenating members in the right order.
        """
        return self.word

class Pad_sample(Sample):
    """
    A dummy sample used for padding
    """
    def __init__(self):
        Sample.__init__(self, word = 0)

def pad_sequences(sequences, pad_func, maxlen = None):
    """
    Similar to keras.preprocessing.sequence.pad_sequence but using Sample as higher level
    abstraction.
    pad_func is a pad class generator.
    """
    ret = []

    # Determine the maxlen
    max_value = max(map(len, sequences))
    if maxlen is None:
        maxlen = max_value
        logging.debug("Padding to maximum observed length ({})".format(max_value))
    else:
        logging.debug("Padding / truncating to {} words (max observed was {})".format(maxlen, max_value))

        # Pad / truncate (done this way to deal with np.array)
    for sequence in sequences:
        cur_seq = list(sequence[:maxlen])
        cur_seq.extend([pad_func()] * (maxlen - len(sequence)))
        ret.append(cur_seq)
    return ret



def load_pretrained_rnn(model_dir):
    """ Static trained model loader function """
    rnn_params = json.load(open(os.path.join(model_dir,
                                             "./model.json")))["rnn"]

    logging.info("Loading model from: {}".format(model_dir))
    rnn = RNN_model(model_fn = RNN_model.set_model_from_file,
                    model_dir = model_dir,
                    **rnn_params)

    # Compile model
    rnn.model_fn()

    return rnn


# Helper functions

## Argmaxes
am = lambda myList: [i[0] for i in sorted(enumerate(myList), key=lambda x:x[1], reverse= True)]

if __name__ == "__main__":
    from pprint import pprint
    args = docopt(__doc__)
    logging.debug(args)
    test_fn = args["--test"]

    if args["--glove"] is not None:
        train_fn = args["--train"]
        dev_fn = args["--dev"]
        epochs = int(args["--epochs"])
        emb_filename = args["--glove"]
        model_dir = "../models/rnn_{}_epocs_{}/".format(epochs,
                                                 emb_filename.split('/')[-1].split(".")[0])
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        rnn = RNN_model(model_fn = RNN_model.set_vanilla_model,
                        sent_maxlen = 20,
                        hidden_units = pow(2, 10),
                        num_of_latent_layers = 2,
                        emb_filename = emb_filename,
                        epochs = epochs,
                        trainable_emb = True,
                        batch_size = 50,
                        model_dir = model_dir)

        rnn.train(train_fn, dev_fn)

    if args["--pretrained"] is not None:
        rnn = load_pretrained_rnn(args["--pretrained"])
        Y, y, metrics = rnn.test(test_fn,
                                 eval_metrics = [("F1 (micro)",
                                                  lambda Y, y: metrics.f1_score(Y, y,
                                                                                average = 'micro')),
                                                 ("Precision (micro)",
                                                  lambda Y, y: metrics.precision_score(Y, y,
                                                                                       average = 'micro')),
                                                 ("Recall (micro)",
                                                  lambda Y, y: metrics.recall_score(Y, y,
                                                                                    average = 'micro')),
                                                 ("Accuracy", metrics.accuracy_score),
                                             ])
"""
- the sentence max length is an important factor on convergence.
This makes sense, shorter sentences are easier to memorize.
The model was able to crack 20 words sentences pretty easily, but seems to be having a harder time with
40 words sentences.
Need to find a better balance.

- relu activation seems to have a lot of positive impact

- The batch size also seems to be having a lot of effect, but I'm not sure how to account for that.

- Maybe actually *increasing* dimensionalty would be better?
There are many ways to crack the train set - we want the model to be free in more
dimensions to allow for more flexibility while still fitting training data.

Ideas:

- test performance on arguments vs. adjuncts.

"""
