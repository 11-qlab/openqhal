# Build the OpenQHAL paper.
# Requires: pdflatex, bibtex (or biber), revtex4-2 class.

PAPER = openqhal
LATEX = pdflatex
BIB   = bibtex

all: $(PAPER).pdf

$(PAPER).pdf: $(PAPER).tex references.bib
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(BIB) $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)
	$(LATEX) -interaction=nonstopmode $(PAPER)

clean:
	rm -f *.aux *.log *.bbl *.blg *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(PAPER).pdf

view: $(PAPER).pdf
	xdg-open $(PAPER).pdf &

.PHONY: all clean distclean view
