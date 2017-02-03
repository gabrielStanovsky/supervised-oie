#!/bin/bash
set -e
echo "News in domain..."
python ./trained_oie_extractor.py \
    --model=../best_models/newswire/rnn_42_epocs_glove \
    --in=../supervised-oie-benchmark/raw_sentences/newswire/propbank.test.txt \
    --out=../evaluations/conlls/newswire_in_domain.conll \
    --conll

# echo "News out of domain..."
# python ./trained_oie_extractor.py \
#     --model=../best_models/newswire \
#     --in=../supervised-oie-benchmark/raw_sentences/wiki/wiki1.test.txt \
#     --out=../evaluations/conlls/newswire_out_of_domain.conll \
#     --conll

# echo "Wiki in domain..."
# python ./trained_oie_extractor.py \
#     --model=../best_models/wiki \
#     --in=../supervised-oie-benchmark/raw_sentences/wiki/wiki1.test.txt \
#     --out=../evaluations/conlls/wiki_in_domain.conll \
#     --conll

# echo "Wiki out of domain..."
# python ./trained_oie_extractor.py \
#     --model=../best_models/wiki \
#     --in=../supervised-oie-benchmark/raw_sentences/newswire/propbank.test.txt \
#     --out=../evaluations/conlls/wiki_out_of_domain.conll \
#     --conll

# echo "Joint..."
# python ./trained_oie_extractor.py \
#     --model=../best_models/joint/ \
#     --in=../supervised-oie-benchmark/raw_sentences/test.txt \
#     --out=../evaluations/conlls/joint.conll \
#     --conll


