set -e
mkdir -p ./eval/test/newswire/
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/Stanford.dat --stanford=./systems_output/test/newswire/stanford_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/OLLIE.dat --ollie=./systems_output/test/newswire/ollie_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/ReVerb.dat --reverb=./systems_output/test/newswire/reverb_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/ClausIE.dat --clausie=./systems_output/test/newswire/clausie_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/OpenIE-4.dat --openiefour=./systems_output/test/newswire/openie4_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/PropS.dat --props=./systems_output/test/newswire/props_propbank_test.txt
python benchmark.py --gold=./oie_corpus/newswire/test.oie.orig --out=eval/test/newswire/rnnie.dat --props=./systems_output/test/newswire/rnnie_propbank_test.txt
python pr_plot.py --in=./eval/test/newswire --out=./eval/test/newswire/eval.png
echo "DONE"
