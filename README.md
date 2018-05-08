<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [supervised-oie](#supervised-oie)
  - [Citing](#citing)
  - [TODO:](#todo)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# supervised-oie
Code for training a supervised Neural Open IE model, as described in our [NAACL2018 paper](https://www.cs.bgu.ac.il/~gabriels/naacl2018.pdf).

Citing
------
If you use this software, please cite:
```
@InProceedings{Stanovsky2018NAACL,
  author    = {Gabriel Stanovsky and Julian Michael and Luke Zettlemoyer and Ido Dagan},
  title     = {Supervised Open Information Extraction},
  booktitle = {Proceedings of The 16th Annual Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (NAACL HLT)},
  month     = {June},
  year      = {2018},
  address   = {New Orleans, Louisiana},
  publisher = {Association for Computational Linguistics},
  pages     = {(to appear)},
}
```

## TODO:
* Make sure to delete [oie_benchmark](oie_benchmark)
python ./rnn/model.py --train=../data/newswire/propbank.train.oie.conll  --test=../data/newswire/propbank.dev.oie.conll --glove=../pretrained_word_embeddings/glove.6B.50d.txt
