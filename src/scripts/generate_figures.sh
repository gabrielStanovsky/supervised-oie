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
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/rnnie_out_of_domain.dat --tabbed=../evaluations/extractions/wiki_out_of_domain.txt
python pr_plot.py --in=${DIR} --out=${DIR}/newswire.png

# Wiki figure
DIR="../evaluations/figures/wiki"
echo "Creating WIKI figure in ${DIR}..."
mkdir -p $DIR
rm -f $DIR/*
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/wiki/clausie_wiki_test.txt
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/wiki/openie4_wiki_test.txt
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/PropS.dat --tabbed=./systems_output/test/wiki/props_wiki_test.txt
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/rnnie_in_domain.dat --tabbed=../evaluations/extractions/wiki_in_domain.txt
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/rnnie_out_of_domain.dat --tabbed=../evaluations/extractions/newswire_out_of_domain.txt
python pr_plot.py --in=${DIR} --out=${DIR}/wiki.png


# Joint figure
# Wiki figure
DIR="../evaluations/figures/joint"
echo "Creating JOINT figure in ${DIR}..."
mkdir -p $DIR
rm -f $DIR/*
python benchmark.py --gold=./oie_corpus/test.oie.orig\
       --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/clausie_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig\
       --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/openie4_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig\
       --out=$DIR/PropS.dat --tabbed=./systems_output/test/props_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig\
       --out=$DIR/rnnie_in_domain.dat --tabbed=../evaluations/extractions/joint.txt
python pr_plot.py --in=${DIR} --out=${DIR}/joint.png


echo "DONE!"
popd
