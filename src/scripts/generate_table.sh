#!/bin/bash
# Usage:
#  ./generate_table.sh
#
# Running this script should reproduce all the numbers from the table in the paper.
set -e
pushd ../supervised-oie-benchmark/

# OIE2016
echo "---- OIE2016 ----"

echo "RNN-aw"
python benchmark.py --gold=./oie_corpus/test.oie.orig   --out=/dev/null  --tabbed=../evaluations/extractions/clean_3.txt

echo "RNN-verb"
python benchmark.py --gold=./oie_corpus/test.oie.orig   --out=/dev/null  --tabbed=../evaluations/extractions/noisy.txt

echo "OpenIE4"
python benchmark.py --gold=./oie_corpus/test.oie.orig   --out=/dev/null  --tabbed=./systems_output/openie4_tabbed.txt

echo "PropS"
python benchmark.py --gold=./oie_corpus/test.oie.orig   --out=/dev/null  --tabbed=./systems_output/props_tabbed.txt

echo "ClausIE"
python benchmark.py --gold=./oie_corpus/test.oie.orig   --out=/dev/null  --tabbed=./systems_output/clausie_tabbed.txt



# WEB
echo "---- WEB ----"

echo "RNN-aw"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie   --out=/dev/null  --tabbed=../evaluations/extractions/web_clean_3.filtered.txt

echo "RNN-verb"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie   --out=/dev/null  --tabbed=../evaluations/extractions/web_clean_3.txt

echo "OpenIE4"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie   --out=/dev/null  --tabbed=./systems_output/openie4_web.txt

echo "PropS"

echo "ClausIE"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/web.oie   --out=/dev/null  --tabbed=./systems_output/clausie_web.txt


# NYT
echo "---- NYT ----"

echo "RNN-aw"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie   --out=/dev/null  --tabbed=../evaluations/extractions/nyt_clean_3.txt

echo "RNN-verb"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie   --out=/dev/null  --tabbed=../evaluations/extractions/nyt_noisy_3.txt

echo "OpenIE4"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie   --out=/dev/null  --tabbed=./systems_output/openie4_nyt.txt

echo "PropS"

echo "ClausIE"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/nyt.oie   --out=/dev/null  --tabbed=./systems_output/clausie_nyt.txt


# PENN

echo "---- PENN ----"

echo "RNN-aw"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie   --out=/dev/null  --tabbed=../evaluations/extractions/penn_clean_3.txt

echo "RNN-verb"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie   --out=/dev/null  --tabbed=../evaluations/extractions/penn_noisy_3.txt

echo "OpenIE4"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie   --out=/dev/null  --tabbed=./systems_output/openie4_penn.txt

echo "PropS"

echo "ClausIE"
python benchmark.py --gold=../external_datasets/mesquita_2013/processed/penn.oie   --out=/dev/null  --tabbed=./systems_output/clausie_penn.txt

# Done
echo "DONE!"
popd
