# Google Colab notebook

Open `SG_DMNMF_Colab.ipynb` in <https://colab.research.google.com>.

## Getting the project into Colab
1. Zip the `sg-dmnmf` folder on your computer.
2. In the notebook, run **Option B** (upload zip) and let it unpack to `/content/sg-dmnmf`.
   Alternatively, copy the folder to Google Drive and run **Option A** instead.
3. Run the remaining cells in order. They install `requirements.txt`, run the Karate sanity
   check, reproduce `experiments/run_all.py --quick`, display the generated tables and
   show the E2/E3 figures.

The notebook reads and runs the existing project; it does not modify the algorithm or the
results. The written report is in the `overleaf/` folder.
