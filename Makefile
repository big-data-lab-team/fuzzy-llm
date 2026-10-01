MAINDOC = paper/main.tex
PDF    = $(MAINDOC:.tex=.pdf)

.PHONY: all pdf clean force cleanall

all: pdf

pdf:
	latexmk -cd -pdf $(MAINDOC)

force: clean pdf

clean:
	latexmk -cd -c $(MAINDOC)
	@rm -f paper/main.bbl paper/main.blg
	@echo "Cleaned auxiliary files for $(MAINDOC)"

cleanall:
	latexmk -cd -c $(MAINDOC)
	rm -f $(PDF) arxiv.tar.gz
	@echo "Cleaned all generated files for $(MAINDOC)"

ARXIV_PAPER_FILES = $(notdir $(wildcard paper/*.tex)) jmlr2e.sty references.bib
ARXIV_FIGURES     = $(wildcard figures/*.pdf)

.PHONY: view arxiv
view: pdf
	@xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || echo "Open $(PDF) manually"

arxiv: pdf
	@rm -f arxiv.tar.gz
	tar -czf arxiv.tar.gz -C paper $(ARXIV_PAPER_FILES) -C .. $(ARXIV_FIGURES)
	@echo "Created arxiv.tar.gz for arXiv submission"
