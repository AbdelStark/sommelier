all: main.pdf

main.pdf: main.tex references.bib figures/fig_results.pdf figures/fig_language_gap.pdf
	tectonic main.tex

.PHONY: figures
figures:
	cd figures && uv run --no-project --with matplotlib python gen_fig_results.py
	cd figures && uv run --no-project --with matplotlib python gen_fig_language_gap.py

.PHONY: clean
clean:
	$(RM) main.pdf main.bbl main.aux main.log main.blg main.out main.synctex.gz
