#!/bin/sh
# Converts all of Mesquita et al.'s files to our format
set -e
python ./scripts/convert_from_mesquita.py --in=../external_datasets/mesquita_2013/experiments/binary/manual/test/penn-ground-truth.txt   --out=../external_datasets/mesquita_2013/processed/penn.oie
cut -f1 ../external_datasets/mesquita_2013/processed/penn.oie | uniq > ../external_datasets/mesquita_2013/processed/penn.raw

python ./scripts/convert_from_mesquita.py --in=../external_datasets/mesquita_2013/experiments/binary/manual/test/web-ground-truth.txt  --out=../external_datasets/mesquita_2013/processed/web.oie
cut -f1 ../external_datasets/mesquita_2013/processed/web.oie | uniq  > ../external_datasets/mesquita_2013/processed/web.raw

python ./scripts/convert_from_mesquita.py --in=../external_datasets/mesquita_2013/experiments/binary/manual/test/nytimes-ground-truth.txt --out=../external_datasets/mesquita_2013/processed/nyt.oie
cut -f1 ../external_datasets/mesquita_2013/processed/nyt.oie | uniq > ../external_datasets/mesquita_2013/processed/nyt.raw

echo "DONE!"
