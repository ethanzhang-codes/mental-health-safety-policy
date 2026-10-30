# Figure 14: Compact Policy Decision Diagram

This is the compact version of the final policy architecture.

- The random forest and safety guardrails receive the available prediction/progression evidence as parallel branches.
- The final level is `max(learned level, guardrail minimum)`.
- Guardrails can raise the predicted level but cannot lower it.
- Actions are offline policy labels; the system does not initiate real emergency responses.

Compile `fig14_policy_decision_compact.tex` with pdfLaTeX in Overleaf, or open the SVG directly in a browser or editor preview.
