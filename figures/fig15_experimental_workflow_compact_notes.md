# Figure 15: Compact Experimental Workflow

The diagram separates model development from held-out reporting:

1. Acquire AnnoMI and CRADLE-Dialogue.
2. Preprocess and retain fixed splits.
3. Train using the training split.
4. Select checkpoints and thresholds using validation data.
5. Develop and select policy/guardrail configurations using training and validation data.
6. Report the frozen selection on test data.

The figure describes the stated selection protocol for the final Confirm and guardrail decisions. The broader repository contains exploratory test reports, so it should not be presented as evidence that no test result was viewed anywhere during project development.

Compile the `.tex` file with pdfLaTeX in Overleaf, or open the SVG directly.
