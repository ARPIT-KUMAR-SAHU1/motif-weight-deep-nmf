# Google Colab notebook

## Open the notebook in Colab

Use this link (opens the notebook directly from the GitHub branch):

**https://colab.research.google.com/github/ARPIT-KUMAR-SAHU1/motif-weight-deep-nmf/blob/feature/complete-project/colab/SG_DMNMF_Colab.ipynb**

The old Google Drive link pointed to an empty notebook with one blank cell; it contains no
work to preserve. The link above opens the complete notebook.

## What it does

The first code cell clones `feature/complete-project` from GitHub into `/content/sg-dmnmf`
(a temporary folder on your own computer when not in Colab). The remaining cells:

1. install `requirements.txt`;
2. run the Karate sanity check (the deep model must match SymNMF);
3. run `python experiments/run_all.py --quick`;
4. display the E1 benchmark table and the E2 alpha-sensitivity values;
5. display the generated matplotlib figures.

All seven code cells were executed successfully with Jupyter before publishing, with no
error outputs. The notebook only reads/runs the project — it does not change the algorithm
or fabricate results. The written report is in `overleaf/` and the slide deck is in
`presentation/` in the same repository.
