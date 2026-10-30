# Figure 13: Final Model Architecture

Files:

- `fig13_final_model_architecture.tex`: self-contained TikZ source for Overleaf or a local TeX installation.
- `fig13_final_model_architecture.svg`: immediately viewable vector version.

The figure reflects the selected research pipeline rather than a generic Transformer:

- current-turn DeBERTa-v3-large Confirm and category branches;
- frozen DeBERTa turn embeddings followed by the selected temporal Alert Transformer;
- causal progression features;
- the random-forest learned policy;
- safety guardrails combined as `max(learned level, guardrail minimum)`;
- four offline escalation outputs and template rationales; and
- a dashed uncertainty branch showing an implemented but unselected ablation.

To compile the TikZ version in Overleaf, upload the `.tex` file and use pdfLaTeX. No external image assets are required.
