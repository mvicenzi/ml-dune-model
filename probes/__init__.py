"""Evaluation suite — the standard metrics for comparing trainings.

This package is the home for model evaluation. It supersedes the assorted tools
under `dino/diagnostics/`, which are being progressively retired.

Everything after extraction is CPU-only: the probe modules import neither
warpconvnet nor a dataset reader, and read features straight off disk. So the
expensive GPU pass happens once per checkpoint while every metric stays cheap to
re-run and iterate on, even on a login node. `extract_features` is the one member
that needs the model, the dataset readers and a GPU — importing a probe module
never pulls it in.

"""
