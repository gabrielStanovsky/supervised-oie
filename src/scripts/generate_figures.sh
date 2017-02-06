#!/bin/bash
set -e
pushd ../supervised-oie-benchmark/

# Joint figure
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

# Joint figure -- only in test
DIR="../evaluations/only_in_test"
echo "Creating OIT  figure in ${DIR}..."
mkdir -p $DIR
rm -f $DIR/*.dat
python benchmark.py --gold=../evaluations/only_in_test/test.oie.filter\
       --out=$DIR/OIT.dat --tabbed=../evaluations/only_in_test/joint.filter
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
echo "Gold"
python oie_readers/calc_corpus_stats.py --gold=./oie_corpus/test.oie.orig --out=$DIR/gold_stats.txt
echo "DONE!"
popd
