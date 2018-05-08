<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [oie-benchmark](#oie-benchmark)
  - [Citing](#citing)
  - [Requirements](#requirements)
  - [Converting QA-SRL to Open IE](#converting-qa-srl-to-open-ie)
  - [Evaluating an Open IE Extractor](#evaluating-an-open-ie-extractor)
  - [Evaluating Existing Systems](#evaluating-existing-systems)
  - [Plotting](#plotting)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# oie-benchmark

Code for converting QA-SRL annotations to Open-IE extractions and comparing Open-IE parsers against a converted benchmark corpus.
This is an implementation of the algorithms described in our [EMNLP2016 paper] (https://www.cs.bgu.ac.il/~gabriels/emnlp2016.pdf).

Citing
------
If you use this software, please cite:
```
@InProceedings{Stanovsky2016EMNLP,
  author    = {Gabriel Stanovsky and Ido Dagan},
  title     = {Creating a Large Benchmark for Open Information Extraction},
  booktitle = {Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing (EMNLP)},
  month     = {November},
  year      = {2016},
  address   = {Austin, Texas},
  publisher = {Association for Computational Linguistics},
  pages     = {(to appear)},
}
```

Requirements
------------
Python 2.7, tested on Ubuntu.
See required python packages [here](requirements.txt).

Additional help can be found in the [FAQ section](faq.txt).

Converting QA-SRL to Open IE
----------------------------
To run the code, you should first obtain the **full** [QA-SRL corpus](https://dada.cs.washington.edu/qasrl/#dataset) and place it under [QASRL-full](QASRL-full).
After obtaining the QA-SRL corpus, run:
```
./create_oie_corpus.sh
```

If everything runs fine, this should create an Open IE corpus (split between wiki and newswire domain) under [oie_corpus](oie_corpus).

Evaluating an Open IE Extractor
-----------------------------
After converting QA-SRL to Open IE, you can now automatically evaluate your Open-IE system against this corpus.
Currently, we support the following Open IE formats:

* [ClausIE](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/software/clausie/)
* [OLLIE](http://knowitall.github.io/ollie/)
* [OpenIE-4](https://github.com/allenai/openie-standalone)
* [PropS](http://u.cs.biu.ac.il/~stanovg/props.html)
* [ReVerb](http://reverb.cs.washington.edu/)
* [Stanford Open IE](http://nlp.stanford.edu/software/openie.html)

To compare your extractor:

1. Run your extractor over the [raw sentences](raw_sentences) and store the output into "*your_output*.txt"

2. Depending on your output format, you can get a precision-recall curve by running [benchmark.py](benchmark.py):
``` 
Usage:
   benchmark --gold=GOLD_OIE --out=OUTPUT_FILE (--stanford=STANFORD_OIE | --ollie=OLLIE_OIE |--reverb=REVERB_OIE | --clausie=CLAUSIE_OIE | --openiefour=OPENIEFOUR_OIE | --props=PROPS_OIE)

Options:
  --gold=GOLD_OIE              The gold reference Open IE file (by default, it should be under ./oie_corpus/all.oie).
  --out=OUTPUT_FILE            The output file, into which the precision recall curve will be written.
  --clausie=CLAUSIE_OIE        Read ClausIE format from file CLAUSIE_OIE.
  --ollie=OLLIE_OIE            Read OLLIE format from file OLLIE_OIE.
  --openiefour=OPENIEFOUR_OIE  Read Open IE 4 format from file OPENIEFOUR_OIE.
  --props=PROPS_OIE            Read PropS format from file PROPS_OIE
  --reverb=REVERB_OIE          Read ReVerb format from file REVERB_OIE
  --stanford=STANFORD_OIE      Read Stanford format from file STANFORD_OIE
```

Evaluating Existing Systems
---------------------------

In the course of this work we tested the above mentioned Open IE parsers against our benchmark.
We provide the output files (i.e., Open IE extractions) of each of these
systems in [systems_output](systems_output).
You can give each of these files to [benchmark.py](benchmark.py), to
get the corresponding precision recall curve.

For example, to evaluate Stanford Open IE output, run:
```
python benchmark.py --gold=./oie_corpus/all.oie --out=./StanfordPR.dat --stanford=./systems_output/stanford_output.txt
```

Plotting
--------

You can plot together multiple outputs of [benchmark.py](benchmark.py), by using [pr_plot.py](pr_plot.py):

```
Usage:
   pr_plot --in=DIR_NAME --out=OUTPUT_FILENAME 

Options:
  --in=DIR_NAME            Folder in which to search for *.dat files, all of which should be in a P/R column format (outputs from benchmark.py).
  --out=OUTPUT_FILENAME    Output filename, filetype will determine the format. Possible formats: pdf, pgf, png
```

Finally, try running:

```
./eval.sh
```

This will create the [Precision Recall figure](./eval/eval.png) using the output of OIE parsers in [systems_output](systems_output).


