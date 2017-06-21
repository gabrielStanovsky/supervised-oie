#!/bin/bash
## An example of training propbank data (after copied and relabled to the data folder)
set -e
python  ./rnn/confidence_model.py  --train=../data/bios/newswire/propbank.train.bios.conll  --dev=../data/bios/newswire/propbank.dev.bios.conll  --test=../data/bios/newswire/propbank.test.bios.conll --load_hyperparams=../hyerparams/confidence.json
