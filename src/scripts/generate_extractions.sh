#!/bin/bash
set -e
python ./trained_oie_extractor.py --in=../evaluations/conlls/joint.conll --out=../evaluations/extractions/joint.txt
python ./trained_oie_extractor.py --in=../evaluations/conlls/newswire_in_domain.conll --out=../evaluations/extractions/newswire_in_domain.txt
python ./trained_oie_extractor.py --in=../evaluations/conlls/newswire_out_of_domain.conll --out=../evaluations/extractions/newswire_out_of_domain.txt
python ./trained_oie_extractor.py --in=../evaluations/conlls/wiki_in_domain.conll --out=../evaluations/extractions/wiki_in_domain.txt
python ./trained_oie_extractor.py --in=../evaluations/conlls/wiki_out_of_domain.conll --out=../evaluations/extractions/wiki_out_of_domain.txt
