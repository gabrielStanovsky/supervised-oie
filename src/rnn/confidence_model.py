""" Usage:
    confidence_model [--train=TRAIN_FN] [--dev=DEV_FN] --test=TEST_FN [--pretrained=MODEL_DIR] [--load_hyperparams=MODEL_JSON] [--saveto=MODEL_DIR]
"""
import numpy as np
import math
import pandas
import nltk
import time
from docopt import docopt
from keras.models import Sequential, Model
from keras.layers import Input, Dense, LSTM, Embedding, TimeDistributedDense,\
    TimeDistributed, merge, Bidirectional, Dropout
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
from pprint import pformat
from common.symbols import NLTK_POS_TAGS
from collections import defaultdict
from rnn.model import Sample, pad_sequences, Pad_sample

import os
import json
from keras.models import model_from_json
import logging
logging.basicConfig(level = logging.DEBUG)


class Confidence_model:
    """
    Represents an RNN model for computing the confidence of an extraction
    """
    def __init__(self,  model_fn, sent_maxlen = None, emb_filename = None,
                 batch_size = 5, seed = 42, sep = '\t',
                 hidden_units = pow(2, 7),trainable_emb = True,
                 emb_dropout = 0.1, num_of_latent_layers = 2,
                 epochs = 10, pred_dropout = 0.1, model_dir = "./models/",
                 pos_tag_embedding_size = 5,
    ):
        """
        Initialize the model
        model_fn - a model generating function, to be called when
                   training with self as a single argument.
        sent_maxlen - the maximum length in words of each sentence -
                      will be used for padding / truncating
        batch_size - batch size for training
        seed - the random seed for reproduciblity
        sep  - separator in the csv dataset files for this model
        hidden_units - number of hidden units per layer
        trainable_emb - controls if the loss should propagate to the word embeddings during training
        emb_dropout - the percentage of dropout during embedding
        num_of_latent_layers - how many LSTMs to stack
        epochs - the number of epochs to train the model
        pred_dropout - the proportion to dropout before prediction
        model_dir - the path in which to save the model
        pos_tag_embedding_size - The number of features to use when encoding pos tags
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
        self.pos_tag_embedding_size = pos_tag_embedding_size

        np.random.seed(self.seed)

        # TODO: this is not needed for confidence, which calculates a real value
        self.num_of_classes = lambda : 1
        self.classes_ = lambda : None


    def confidence_prediction(self, inputs_and_embeddings):
        """
        Return a network computing confidence of the given OIE inputs
        """
        return predict_layer(dropout(latent_layers(merge([embed(inp)
                                                          for inp, embed in inputs_and_embeddings],
                                                         mode = "concat",
                                                         concat_axis = -1))))


    # TODO: these should probably be deleted

    def transform_labels(self, labels):
        """
        Encode a list of textual labels
        """
        # Fallback:
        # return self.encoder.transform(labels)
        classes  = list(self.classes_())
        return [classes.index(label) for label in labels]

    # TODO: put all of the functions below in a super class (Functional_keras_model, Functional_sentenial_model)
    # General utils

    def plot(self, fn, train_fn):
        """
        Plot this model to an image file
        Train file is needed as it influences the dimentions of the RNN
        """
        from keras.utils.visualize_util import plot
        X, Y = self.load_dataset(train_fn)
        self.model_fn()
        plot(self.model, to_file = fn)

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
        word_inputs = []
        pred_inputs = []
        pos_inputs = []
        sents = self.get_fixed_size(sents)

        for sent in sents:
            # pd assigns NaN for very infreq. empty string (see wiki train)
            sent_words = [word
                         if not (isinstance(word, float) and math.isnan(word)) else " "
                         for word in sent.word.values]

            pos_tags_encodings = [NLTK_POS_TAGS.index(tag)
                                  for (_, tag)
                                  in nltk.pos_tag(sent_words)]
            word_encodings = [self.emb.get_word_index(w) for w in sent_words]
            pred_word_encodings = [self.emb.get_word_index(w) for w in sent_words]
            word_inputs.append([Sample(w) for w in word_encodings])
            pred_inputs.append([Sample(w) for w in pred_word_encodings])
            pos_inputs.append([Sample(pos) for pos in pos_tags_encodings])

        # Pad / truncate to desired maximum length
        ret = {"word_inputs" : [],
               "predicate_inputs": []}
        ret = defaultdict(lambda: [])

        for name, sequence in zip(["word_inputs", "predicate_inputs", "postags_inputs"],
                                  [word_inputs, pred_inputs, pos_inputs]):
            for samples in pad_sequences(sequence,
                                         pad_func = lambda : Pad_sample(),
                                         maxlen = self.sent_maxlen):
                ret[name].append([sample.encode() for sample in samples])

        return {k: np.array(v) for k, v in ret.iteritems()}

    def get_fixed_size(self, sents):
        """
        Partition sents into lists of sent_maxlen elements
        (execept the last in each sentence, which might be shorter)
        """
        return [sent[s_ind : s_ind + self.sent_maxlen]
                for sent in sents
                for s_ind in range(0, len(sent), self.sent_maxlen)]




    def encode_outputs(self, sents):
        """
        Given a dataframe split to sentences, encode outputs for rnn classification.
        Should return a list sequence of sample of length maxlen.
        """
        output_encodings = []
        sents = self.get_fixed_size(sents)
        # Encode outputs
        for sent in sents:
            output_encodings.append(list(np_utils.to_categorical(\
                                                list(self.transform_labels(sent.label.values)),
                                                            nb_classes = self.num_of_classes())))

        # Pad / truncate to maximum length
        return np.ndarray(shape = (len(sents),
                                  self.sent_maxlen,
                                  self.num_of_classes()),
                          buffer = np.array(pad_sequences(output_encodings,
                                                          lambda : \
                                                            np.zeros(self.num_of_classes()),
                                                          maxlen = self.sent_maxlen)))



    # Functional Keras -- all of the following are currying functions expecting models as input
    # https://keras.io/getting-started/functional-api-guide/

    def embed_word(self):
        """
        Embed word sequences using self's embedding class
        """
        return self.emb.get_keras_embedding(dropout = self.emb_dropout,
                                            trainable = self.trainable_emb,
                                            input_length = self.sent_maxlen)

    def embed_pos(self):
        """
        Embed Part of Speech using this instance params
        """
        return Embedding(output_dim = self.pos_tag_embedding_size,
                         input_dim = len(NLTK_POS_TAGS),
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


    def set_model(self):
        """
        Set a Keras model for predicting OIE as a member of this class
        Can be passed as model_fn to the constructor
        """
        logging.debug("Setting vanilla model")
        # Build model

        ## Embedding Layer
        word_embedding_layer = self.embed_word()
        pos_embedding_layer = self.embed_pos()
#        label_embedding_layer = self.embed_label()

        ## Deep layers
        latent_layers = self.stack_latent_layers(self.num_of_latent_layers)

        ## Dropout
        dropout = Dropout(self.pred_dropout)

        ## Prediction
        predict_layer = self.predict_classes()

        ## Prepare input features, and indicate how to embed them

        # True input
        true_input = [(Input(shape = (self.sent_maxlen,),
                             dtype="int32",
                             name = "word_inputs"),
                       word_embedding_layer),
                      (Input(shape = (self.sent_maxlen,),
                             dtype="int32",
                             name = "postags_inputs"),
                       pos_embedding_layer)
        ]

        corrupt_input = [(Input(shape = (self.sent_maxlen,),
                             dtype="int32",
                             name = "neg_word_inputs"),
                          word_embedding_layer),
                         (Input(shape = (self.sent_maxlen,),
                                dtype="int32",
                                name = "neg_postags_inputs"),
                          pos_embedding_layer)]


        # true_input = [(Input(shape = (self.sent_maxlen,),
        #                      dtype="int32",
        #                      name = "word_inputs"),
        #                word_embedding_layer),
        #               (Input(shape = (self.sent_maxlen,),
        #                      dtype="int32",
        #                      name = "predicate_inputs"),
        #                word_embedding_layer),
        #               (Input(shape = (self.sent_maxlen,),
        #                      dtype="int32",
        #                      name = "postags_inputs"),
        #                pos_embedding_layer),
        #               (Input(shape = (self.sent_maxlen,),
        #                      dtype="int32",
        #                      name = "postags_inputs"),
        #                label_embedding_layer),
        # ]

        # # Corrput negative sample
        # corrupt_input = [(Input(shape = (self.sent_maxlen,),
        #                                 dtype="int32",
        #                         name = "neg_word_inputs"),
        #                   word_embedding_layer),
        #                  (Input(shape = (self.sent_maxlen,),
        #                         dtype="int32",
        #                         name = "neg_predicate_inputs"),
        #                   word_embedding_layer),
        #                  (Input(shape = (self.sent_maxlen,),
        #                         dtype="int32",
        #                         name = "neg_postags_inputs"),
        #                   pos_embedding_layer),
        #                  (Input(shape = (self.sent_maxlen,),
        #                         dtype="int32",
        #                         name = "neg_postags_inputs"),
        #                   label_embedding_layer),
        # ]

        confidence_prediction = lambda inputs_and_embeddings:\
                                predict_layer(dropout(latent_layers(merge([embed(inp)
                                                                           for inp, embed in inputs_and_embeddings],
                                                                          mode = "concat",
                                                                        concat_axis = -1))))



        # Compute two "branches" for confidence estimation - one true and one corrput
        true_confidence = confidence_prediction(true_input)
        neg_confidence = confidence_prediction(corrupt_input)

        # Combine these
        output = merge([true_confidence, neg_confidence],
                       mode = "sum")

        # Build model
        self.model = Model(input = map(itemgetter(0), true_input + corrupt_input),
                           output = [output])

        # Loss
        self.model.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['categorical_accuracy'])
        self.model.summary()

        # Save model json to file
        self.save_model_to_file(os.path.join(self.model_dir, "confidence_model.json"))

    def to_json(self):
        """
        Encode a json of the parameters needed to reload this model
        """
        return {
            "sent_maxlen": self.sent_maxlen,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "sep": self.sep,
            "hidden_units": self.hidden_units,
            "trainable_emb": self.trainable_emb,
            "emb_dropout": self.emb_dropout,
            "num_of_latent_layers": self.num_of_latent_layers,
            "epochs": self.epochs,
            "pred_dropout": self.pred_dropout,
            "emb_filename": self.emb_filename,
            "pos_tag_embedding_size": self.pos_tag_embedding_size,
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



if __name__ == "__main__":
    from pprint import pprint
    args = docopt(__doc__)
    logging.debug(args)
    test_fn = args["--test"]

    if args["--train"] is not None:
        train_fn = args["--train"]
        dev_fn = args["--dev"]

        if args["--load_hyperparams"] is not None:
            # load hyperparams from json file
            json_fn = args["--load_hyperparams"]
            logging.info("Loading model from: {}".format(json_fn))
            rnn_params = json.load(open(json_fn))["rnn"]

        else:
            # Use some default params
            rnn_params = {"sent_maxlen":  20,
                          "hidden_units": pow(2, 10),
                          "num_of_latent_layers": 2,
                          "emb_filename": emb_filename,
                          "epochs": 10,
                          "trainable_emb": True,
                          "batch_size": 50,
                          "emb_filename": "../pretrained_word_embeddings/glove.6B.50d.txt",
            }


        logging.debug("hyperparams:\n{}".format(pformat(rnn_params)))
        if args["--saveto"] is not None:
            model_dir = os.path.join(args["--saveto"], "{}/".format(time.strftime("%d_%m_%Y_%H_%M")))
        else:
            model_dir = "../models/{}/".format(time.strftime("%d_%m_%Y_%H_%M"))
        logging.debug("Saving models to: {}".format(model_dir))
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        rnn = Confidence_model(model_fn = Confidence_model.set_model,
                               model_dir = model_dir,
                               **rnn_params)
        rnn.set_model()
        rnn.plot("./confidence_model.jpg", train_fn)
