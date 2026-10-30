# Contributing

Contributions should preserve the project's privacy boundaries:

- do not commit raw conversations or linkable case identifiers;
- do not commit model checkpoints, embeddings, or large generated outputs;
- report aggregate metrics with clear split and selection provenance;
- keep validation-based selection separate from held-out test reporting; and
- describe the system as research software, not a clinically validated product.

Before opening a pull request, run a secret scan and inspect the staged file list:

```bash
git status --short
git diff --cached --stat
```

