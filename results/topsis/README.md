# TOPSIS: nine benchmark conditions

One independent eight-model decision matrix per condition, using (1) macro-F1 percent (maximize), (2) amortized pipeline E2E latency in ms/image (minimize), (3) deployment artifact size in MiB (minimize). Each matrix is computed at batch size 1 from the seed 42 model.

Conditions: `TRT_FP32`, `TRT_FP16`, `TRT_INT8`, `PYTORCH_FP32`, `PYTORCH_FP16`, `JCPU_FP32`, `JCPU_INT8`, `P5CPU_FP32`, `P5CPU_INT8`.

Files:
- `decision_matrices.csv` — source metrics for each 8×3 matrix;
- `weights.csv` — Equal, Entropy and CRITIC weights per condition;
- `scores_and_ranks.csv` — TOPSIS closeness coefficients and ranks;
- `mean_ranks_9_conditions.csv` — mean model rank across all nine conditions for each weight method.

`python scripts/verify_topsis.py` independently recomputes all weights and scores from the decision matrices and checks published values. For PyTorch precision choices, checkpoint *storage* size remains FP32 in both cases; this is not GPU memory usage.
