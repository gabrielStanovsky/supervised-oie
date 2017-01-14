#!/bin/bash
set -e
# Copy the produced conll files to the data folder
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.dev.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.train.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.test.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.dev.oie.conll ../data/wiki/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.train.oie.conll ../data/wiki/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.test.oie.conll ../data/wiki/

# Relabel for better distribution

# propbank
echo "Relabeling PB..."
python ./scripts/relabel.py --in=../data/newswire/propbank.dev.oie.conll\
       --out=../data/bios/newswire/propbank.dev.bios.conll
python ./scripts/relabel.py --in=../data/newswire/propbank.train.oie.conll\
       --out=../data/bios/newswire/propbank.train.bios.conll
python ./scripts/relabel.py --in=../data/newswire/propbank.test.oie.conll\
       --out=../data/bios/newswire/propbank.test.bios.conll

# wiki
echo "Relabeling Wiki"
python ./scripts/relabel.py --in=../data/wiki/wiki1.dev.oie.conll\
       --out=../data/bios/wiki/wiki1.dev.bios.conll
python ./scripts/relabel.py --in=../data/wiki/wiki1.train.oie.conll\
       --out=../data/bios/wiki/wiki1.train.bios.conll
python ./scripts/relabel.py --in=../data/wiki/wiki1.test.oie.conll\
       --out=../data/bios/wiki/wiki1.test.bios.conll

## TODO: These are buggy, as they are made from concatenating, messying up unique sentence sentences
#cp -v ../supervised-oie-benchmark/oie_corpus/dev.oie.conll ../data/
#cp -v ../supervised-oie-benchmark/oie_corpus/train.oie.conll ../data/
#cp -v ../supervised-oie-benchmark/oie_corpus/test.oie.conll ../data/
#cp -v ../supervised-oie-benchmark/oie_corpus/all.oie.conll ../data


echo "DONE!"
