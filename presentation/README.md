# SG-DMNMF presentation

Open **`SG_DMNMF_Presentation_Full.pptx`** — this is the new, visual, self-explanatory deck.
(The older `SG_DMNMF_Presentation.pptx` is the short first version; it may be open in
PowerPoint, which is why the new file has a different name.)

- **28 slides**, 16:9, clean academic style.
- **18 diagrams/charts** and **2 result tables** — network diagrams, triangle motif example,
  matrix heat maps, deep-NMF architecture, perturbation illustration, AMI explanation,
  full pipeline, experiment map, noise types, dashboard layout schematic, and charts taken
  from the actual CSV results.
- **Speaker notes on every slide** with the exact line to say. Each slide also contains full
  sentences, so you can read directly from the screen without improvising.

## Suggested 15–20 minute flow
1. Slides 1–2: title and roadmap.
2. Slides 3–7: background, problem, gap, research questions, contribution.
3. Slides 8–13: the four method steps, AMI, complete algorithm, alpha decision.
4. Slides 14–23: experiment map, setup, E1–E4 charts and findings.
5. Slides 24–28: RQ answers, limitations, dashboard deliverables, conclusion, questions.

If the teacher asks about exact paper reproduction, show slide 25: this is the symmetric,
triangle-only **DMRA-lite** simplification; the contribution is the label-free alpha selector.

## Regenerating
Run `python build_presentation.py` from the project root (needs `python-pptx`, matplotlib,
networkx). All numbers and charts are read from `results/*.csv`; nothing is invented.
Charts and diagrams are also stored in `assets/`.
