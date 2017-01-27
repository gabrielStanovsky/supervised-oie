#!/bin/bash
set -e
echo "Converting to tabbed format"
echo "ClausIE.."
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=clausie\
       --in=./systems_output/clausie_output.txt\
       --out=./systems_output/clausie_tabbed.txt

echo "OLLIE.."
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=ollie\
       --in=./systems_output/ollie_output.txt\
       --out=./systems_output/ollie_tabbed.txt

echo "openie4"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=openie4\
       --in=./systems_output/openie4_output.txt\
       --out=./systems_output/openie4_tabbed.txt


echo "props"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=props\
       --in=./systems_output/props_output.txt\
       --out=./systems_output/props_tabbed.txt

echo "reverb"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=reverb\
       --in=./systems_output/reverb_output.txt\
       --out=./systems_output/reverb_tabbed.txt

echo "stanford"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/all.txt\
       --reader=stanford\
       --in=./systems_output/stanford_output.txt\
       --out=./systems_output/stanford_tabbed.txt


echo "Splitting newswire.."
echo "ClausIE.."
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=clausie\
       --in=./systems_output/clausie_output.txt\
       --out=./systems_output/test/newswire/clausie_propbank_test.txt

echo "OLLIE.."
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=ollie\
       --in=./systems_output/ollie_output.txt\
       --out=./systems_output/test/newswire/ollie_propbank_test.txt

echo "openie4"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=openie4\
       --in=./systems_output/openie4_output.txt\
       --out=./systems_output/test/newswire/openie4_propbank_test.txt


echo "props"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=props\
       --in=./systems_output/props_output.txt\
       --out=./systems_output/test/newswire/props_propbank_test.txt

echo "reverb"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=reverb\
       --in=./systems_output/reverb_output.txt\
       --out=./systems_output/test/newswire/reverb_propbank_test.txt

echo "stanford"
python oie_readers/split_corpus.py\
       --corpus=./raw_sentences/newswire/propbank.test.txt\
       --reader=stanford\
       --in=./systems_output/stanford_output.txt\
       --out=./systems_output/test/newswire/stanford_propbank_test.txt

echo "DONE!"
