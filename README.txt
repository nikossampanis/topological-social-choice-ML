ENRICHED Overleaf ZIP contents
==============================

- main.tex                  : enriched journal-draft with full proofs, analyses, APA-style tables
- references.bib            : bibliography
- data/tda_pref_profile_results.csv
- data/tda_pref_profile_results_table.tex
- code/classification_pipeline.py : complete end-to-end pipeline (single file)

Overleaf:
1) Upload ZIP
2) Set main.tex as main file
3) Recompile

Regenerating results:
- Put your .soi election files into ./elections/ (locally)
- Run code/classification_pipeline.py
- Copy generated CSV/TEX into data/ and recompile
