all:
	pdflatex master.tex
	makeindex -s master.ist master.idx
	bibtex master
	pdflatex master.tex
	pdflatex master.tex

clean:
	find . -type f -name '*.aux' -print0 | xargs -0 rm
	rm *.ind *.idx *.ilg *.log *.out *.toc *.bbl *.blg *-blx.bib *.run.xml

view:	
	xdg-open master.pdf &>/dev/null &

commit:
	git commit -a
	git push
