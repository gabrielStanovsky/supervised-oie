<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [supervised-oie](#supervised-oie)
  - [Citing](#citing)
  - [TODO:](#todo)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# supervised-oie
Code for training a supervised Neural Open IE model, as described in our [NAACL2018 paper](https://www.cs.bgu.ac.il/~gabriels/naacl2018.pdf).

Citing
------
If you use this software, please cite:
```
@InProceedings{Stanovsky2018NAACL,
  author    = {Gabriel Stanovsky and Julian Michael and Luke Zettlemoyer and Ido Dagan},
  title     = {Supervised Open Information Extraction},
  booktitle = {Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {(to appear)},
}
```

# Quickstart

1. Install requirements
```bash
pip install requirements.txt
```

2. Download OIE corpus
```bash
cd ./src
./scripts/download_external.sh
```

3. Download Embeddings
```bash
cd ./pretrained_word_embeddings/
./download_external.sh
```

2. Train model
```
cd ./src
python ./rnn/model.py --train=../data/newswire/propbank.train.oie.conll  --test=../data/newswire/propbank.dev.oie.conll --glove=../pretrained_word_embeddings/glove.6B.50d.txt
```


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
