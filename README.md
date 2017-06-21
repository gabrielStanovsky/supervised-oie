# supervised-oie

## External Requirements
* Recurrent shop
* Seq2Seq

## TODO:
* Make sure to delete [oie_benchmark](oie_benchmark)
python ./rnn/model.py --train=../data/newswire/propbank.train.oie.conll  --test=../data/newswire/propbank.dev.oie.conll --glove=../pretrained_word_embeddings/glove.6B.50d.txt
python ./rnn/seq2seq_model.py --train=a --dev=a --test=a --hyperparams=../hyperparams/seq2seq.json --saveto=a
