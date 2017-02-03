#!/bin/bash
set -e
pushd ../supervised-oie-benchmark/
# Newswire figure
DIR="../evaluations/figures/newswire"
echo "Creating Newswire figure in ${DIR}..."
mkdir -p $DIR
rm -f $DIR/*
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/newswire/clausie_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/newswire/openie4_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/PropS.dat --tabbed=./systems_output/test/newswire/props_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/rnnie_in_domain.dat --tabbed=../evaluations/extractions/newswire_in_domain.txt
# python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
#        --out=$DIR/rnnie_out_of_domain.dat --tabbed=../evaluations/extractions/wiki_out_of_domain.txt
python pr_plot.py --in=${DIR} --out=${DIR}/newswire.png

# Wiki figure

# Joint figure

echo "DONE!"
popd
