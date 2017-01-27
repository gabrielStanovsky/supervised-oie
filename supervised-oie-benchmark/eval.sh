#!/bin/bash
mkdir -p ./eval/
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/Stanford.dat --stanford=./systems_output/stanford_output.txt
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/OLLIE.dat --ollie=./systems_output/ollie_output.txt
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/ReVerb.dat --reverb=./systems_output/reverb_output.txt
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/ClausIE.dat --clausie=./systems_output/clausie_output.txt
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/OpenIE-4.dat --openiefour=./systems_output/openie4_output.txt
python benchmark.py --gold=./oie_corpus/all.oie.orig --out=eval/PropS.dat --props=./systems_output/props_output.txt
python pr_plot.py --in=./eval --out=./eval/eval.png
echo "DONE"
