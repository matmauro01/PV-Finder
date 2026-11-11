"""Data/MC overlays and the AMVF-as-reference bias study for note Section 6.2.

Inputs are the April 2026 low-pile-up evaluations
(``outputs/04_16_2026_output/reproduction_ep300_{mc,run2,run3}/eval_results.pkl``);
no inference is run.  The MC AMVF and truth vertices are re-read from the ROOT
ntuple through the same h5 -> ROOT index map the MC eval used, and that map is
verified event by event before any number is taken from it.

Modules
-------
samples    Load the pkls, the MC ROOT vertices, and the historical sigma fit.
reference  Data-style matching with any reference, paired event bootstrap.
plots      Resolution and yield figures with MC overlaid on data.
"""
