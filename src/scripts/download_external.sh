#!/bin/bash
## Download external data required for this project
set -e
# Download and unpack oie corpus
wget https://github.com/gabrielStanovsky/oie-benchmark/blob/master/snapshot_oie_corpus.tar.gz
tar xvzf snapshot_oie_corpus.tar.gz
