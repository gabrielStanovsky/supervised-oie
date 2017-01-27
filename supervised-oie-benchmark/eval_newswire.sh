#!/bin/bash
set -e
mkdir -p ./eval/test/newswire/
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/Stanford.dat --tabbed=./systems_output/test/newswire/stanford_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/OLLIE.dat --tabbed=./systems_output/test/newswire/ollie_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/ReVerb.dat --tabbed=./systems_output/test/newswire/reverb_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/ClausIE.dat --tabbed=./systems_output/test/newswire/clausie_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/OpenIE-4.dat --tabbed=./systems_output/test/newswire/openie4_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/PropS.dat --tabbed=./systems_output/test/newswire/props_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/propbank.test.oie.orig --out=eval/test/newswire/rnnie.dat --tabbed=./systems_output/test/newswire/rnnie_propbank_test.txt
python pr_plot.py --in=./eval/test/newswire --out=./eval/test/newswire/eval.png
echo "DONE"
