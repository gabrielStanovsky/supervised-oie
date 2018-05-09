#!/bin/bash
set -e
pushd ../supervised-oie-benchmark/

# Joint figure
DIR="../evaluations/figures/joint"
echo "Creating JOINT figure in ${DIR}..."
mkdir -p $DIR
rm -f $DIR/auc.dat
python benchmark.py --gold=./oie_corpus/test.oie.orig \
       --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/clausie_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig \
       --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/openie4_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig \
       --out=$DIR/PropS.dat --tabbed=./systems_output/test/props_test.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig \
       --out=$DIR/RnnOIE.dat --tabbed=../evaluations/extractions/joint.txt
python benchmark.py --gold=./oie_corpus/test.oie.orig \
       --out=$DIR/Clean.dat --tabbed=../evaluations/extractions/clean_3.txt
# python benchmark.py --gold=./oie_corpus/test.oie.orig \
#        --out=$DIR/Noisy.dat --tabbed=../evaluations/extractions/noisy.txt
python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png

# # Experimental
# DIR="../evaluations/figures/joint/experimental/"
# echo "Creating EXPERIMENTAL figure in ${DIR}..."
# mkdir -p $DIR
# rm -f $DIR/*
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/clausie_test.txt --exactMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/openie4_test.txt --exactMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/PropS.dat --tabbed=./systems_output/test/props_test.txt --exactMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/RnnOIE.dat --tabbed=../evaluations/extractions/joint.txt --exactMatch
# python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png


# # Argument match
# DIR="../evaluations/figures/joint/arguments/"
# echo "Creating EXPERIMENTAL figure in ${DIR}..."
# mkdir -p $DIR
# rm -f $DIR/*
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/clausie_test.txt --argMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/openie4_test.txt --argMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/PropS.dat --tabbed=./systems_output/test/props_test.txt --argMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/RnnOIE.dat --tabbed=../evaluations/extractions/joint.txt --argMatch
# python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png



# # Predicate Match
# DIR="../evaluations/figures/joint/predicate/"
# echo "Creating PREDICATE MATCH figure in ${DIR}..."
# mkdir -p $DIR
# rm -f $DIR/*
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/ClausIE.dat --tabbed=./systems_output/test/clausie_test.txt --predMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/OpenIE-4.dat --tabbed=./systems_output/test/openie4_test.txt --predMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/PropS.dat --tabbed=./systems_output/test/props_test.txt --predMatch
# python benchmark.py --gold=./oie_corpus/test.oie.orig\
#        --out=$DIR/RnnOIE.dat --tabbed=../evaluations/extractions/joint.txt --predMatch
# python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png


# # Joint figure -- only in test
# DIR="../evaluations/only_in_test"
# echo "Creating OIT  figure in ${DIR}..."
# mkdir -p $DIR
# rm -f $DIR/*.dat
# python benchmark.py --gold=../evaluations/only_in_test/test.oie.oit\
#        --out="$DIR/seen.dat" --tabbed=../evaluations/only_in_test/joint.oit
# python benchmark.py --gold=../evaluations/only_in_test/test.oie.noit\
#        --out="$DIR/unseen.dat" --tabbed=../evaluations/only_in_test/joint.noit
# python pr_plot.py --in=${DIR} --out=${DIR} --outputtype=png

# Calculate stats:
## avg arguments per sentence
## avg words per argument
## average predicate length
## Average proposition length
# DIR="../evaluations/figures/joint/"
# echo "Calculating stats"
# echo "ClausIE"
# python oie_readers/calc_corpus_stats.py --in=./systems_output/test/clausie_test.txt --out=$DIR/clausie_stats.txt
# echo "Open IE4"
# python oie_readers/calc_corpus_stats.py --in=./systems_output/test/openie4_test.txt --out=$DIR/openie4_stats.txt
# echo "PropS"
# python oie_readers/calc_corpus_stats.py --in=./systems_output/test/props_test.txt --out=$DIR/props_stats.txt
# echo "RnnOIE"
# python oie_readers/calc_corpus_stats.py --in=../evaluations/extractions/joint.txt --out=$DIR/rnnoie_stats.txt
# echo "Noisy"
# python oie_readers/calc_corpus_stats.py --in=../evaluations/extractions/noisy.txt --out=$DIR/rnnoie_stats.txt
# echo "Gold"
# python oie_readers/calc_corpus_stats.py --gold=./oie_corpus/test.oie.orig --out=$DIR/gold_stats.txt
echo "DONE!"
popd
