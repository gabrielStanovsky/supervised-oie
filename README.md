# supervised-oie

## External Requirements
* Recurrent shop
* Seq2Seq

## Training

    python ./rnn/model.py --train=../data/newswire/propbank.train.oie.conll  --test=../data/newswire/propbank.dev.oie.conll --glove=../pretrained_word_embeddings/glove.6B.50d.txt
    python ./rnn/seq2seq_model.py --train=../oie_corpus/train.oie  --dev=../oie_corpus/dev.oie  --test=../oie_corpus/test.oie  --hyperparams=../hyperparams/seq2seq.json --saveto=../models/seq2seq/

## TODO:
* Make sure to delete [oie_benchmark](oie_benchmark)

## Time stats (sec / sentence)
Averaged over a run on 3200 sentences, including startup time for all systems.

* RnnOIE - 0.074
* OpenIE 4.0 - 0.065
* ClausIE - 0.246
* PropS - 0.218
* Stanford Parser - 0.2
