#!/bin/bash
set -e
python ./trained_oie_extractor.py --model=../best_models/newswire --in=../supervised-oie-benchmark/raw_sentences/all.txt --out=../evaluations/all_2016.conll --conll > /dev/null
