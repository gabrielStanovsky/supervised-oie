#!/bin/bash
# Copy the produced conll files to the data folder
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.dev.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.train.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/newswire/propbank.test.oie.conll ../data/newswire/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.dev.oie.conll ../data/wiki/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.train.oie.conll ../data/wiki/
cp -v ../supervised-oie-benchmark/oie_corpus/wiki/wiki1.test.oie.conll ../data/wiki/
cp -v ../supervised-oie-benchmark/oie_corpus/dev.oie.conll ../data/
cp -v ../supervised-oie-benchmark/oie_corpus/train.oie.conll ../data/
cp -v ../supervised-oie-benchmark/oie_corpus/test.oie.conll ../data/
