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
       --out=$DIR/RnnOIE_in_domain.dat --tabbed=../evaluations/extractions/newswire_in_domain.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig\
       --out=$DIR/RnnOIE_out_of_domain.dat --tabbed=../evaluations/extractions/wiki_out_of_domain.txt
python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png

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
       --out=$DIR/RnnOIE_in_domain.dat --tabbed=../evaluations/extractions/wiki_in_domain.txt
python benchmark.py --gold=./oie_corpus/wiki/wiki1.test.oie.orig\
       --out=$DIR/RnnOIE_out_of_domain.dat --tabbed=../evaluations/extractions/newswire_out_of_domain.txt
python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png


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
       --out=$DIR/RnnOIE.dat --tabbed=../evaluations/extractions/joint.txt
python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png


DIR="../evaluations/figures/joint/"
echo "Calculating stats"
echo "ClausIE"
python oie_readers/calc_corpus_stats.py --in=./systems_output/test/clausie_test.txt --out=$DIR/clausie_stats.txt
echo "Open IE4"
python oie_readers/calc_corpus_stats.py --in=./systems_output/test/openie4_test.txt --out=$DIR/openie4_stats.txt
echo "PropS"
python oie_readers/calc_corpus_stats.py --in=./systems_output/test/props_test.txt --out=$DIR/props_stats.txt
echo "RnnOIE"
python oie_readers/calc_corpus_stats.py --in=../evaluations/extractions/joint.txt --out=$DIR/rnnoie_stats.txt
echo "DONE!"
popd
