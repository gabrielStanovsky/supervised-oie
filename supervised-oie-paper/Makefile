all:emnlp_main.pdf

TEX=emnlp_main

emnlp_main.pdf:$(TEX)
	pdflatex $<
	bibtex eacl2017
	pdflatex $<
	pdflatex $<


clean:
	rm -f *~ emanlp_main.pdf
