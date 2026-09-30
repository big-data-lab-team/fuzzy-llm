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
	rm -f $(PDF)
	@echo "Cleaned all generated files for $(MAINDOC)"

.PHONY: view
view: pdf
	@xdg-open $(PDF) 2>/dev/null || open $(PDF) 2>/dev/null || echo "Open $(PDF) manually"
