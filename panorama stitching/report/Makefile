all: report clean

report: report.tex report.bib
	pdflatex report.tex && bibtex report && pdflatex report.tex && pdflatex report.tex

.PHONY: clean
clean:
	rm *.aux *.log
