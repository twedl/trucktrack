.PHONY: slides slides-preview clean-slides

slides:
	quarto render slides/

slides-preview:
	quarto preview slides/

clean-slides:
	rm -rf slides/_freeze slides/.quarto docs/index.html docs/index_files
