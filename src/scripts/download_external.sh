#!/bin/bash
## Download external data required for this project
set -e
if [ ! -f "snapshot_oie_corpus.tar.gz" ]
then
    echo "Downloading and unpacking OIE corpus..."
      # Download and unpack oie corpus
    wget https://github.com/gabrielStanovsky/oie-benchmark/raw/master/snapshot_oie_corpus.tar.gz
    tar xvzf snapshot_oie_corpus.tar.gz
    echo "Done!"
fi

# Done
echo "Succesfully finished downloading external data"
