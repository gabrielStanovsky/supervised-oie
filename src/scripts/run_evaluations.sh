#!/bin/bash
set -e
python ./trained_oie_extractor.py --model=../best_models/newswire --in=../supervised-oie-benchmark/raw_sentences/newswire/propbank.test.txt --out=../evaluations/newswire_in_domain.conll --conll
