# Overleaf project (LaTeX report)

## Quick start — paste only `main.tex`

`main.tex` is now **self-contained**. You can create a new Overleaf project, paste its
contents into `main.tex`, and compile with the default **pdfLaTeX** engine. No extra uploads
are required:

- the bibliography is inline (`thebibliography`), so `references.bib` is not needed;
- the three charts are drawn with `pgfplots` from the values in `results/*.csv`, so the PNG
  files in `figures/` are not needed.

The earlier errors ("File `figures/...` not found" and "couldn't open database file
references.bib") came from uploading only `main.tex` while the figure folder and `.bib` file
were missing. That dependency has been removed.

## Optional files
- `references.bib` — the same references in BibTeX format, if you prefer an external
  bibliography.
- `figures/` — copies of the original matplotlib PNGs with LaTeX-safe names, if you want to
  swap the pgfplots charts for the original images.

## Compilation notes
- Compiler: **pdfLaTeX** (Overleaf default). `pgfplots`, `tikz`, `graphicx`, `natbib`,
  `algorithm` and `booktabs` are all standard on Overleaf.
- Result tables are wrapped in `\resizebox` so they fit the page width.
- Any remaining `Overfull \hbox` messages are cosmetic warnings, not errors.

## Honesty note
- The report states the implementation is the symmetric, triangle-only **DMRA-lite**
  simplification and **not** an exact reproduction of DMRA-NMF.
- Offline synthetic datasets are marked as such; the real Dolphin graph stays unlabeled.
- Bibliographic details came from the CP1 literature matrix in
  `Team8_ArpitSahu_06October2026.pdf`; verify them against publishers before submission.
- All numbers come from the committed `results/` CSV files. Re-run the experiments and
  regenerate the tables/charts if the results change.
