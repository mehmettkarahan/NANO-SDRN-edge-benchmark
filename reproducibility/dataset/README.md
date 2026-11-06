## Dataset source

The original real images used in this study were obtained from the publicly
available PlantVillage Dataset repository:

spMohanty/PlantVillage-Dataset
https://github.com/spMohanty/PlantVillage-Dataset

This study used the following three tomato classes:

- Tomato__Early_blight
- Tomato__Late_blight
- Tomato__healthy

The selected real images were organized into the `real_validation` and
`real_test` subsets used in this study. The `synthetic_train` set was generated
separately using the img2img procedure documented under
`reproducibility/synthetic_generation/`.

The exact image lists used in each experimental split are provided in the
corresponding manifest files in this directory.

# Dataset Integrity Audit

This directory documents the dataset-integrity checks used for the synthetic-to-real evaluation protocol.

## Dataset composition

- `synthetic_train`: 6,000 synthetic images (2,000 per class)
- `real_valid`: 1,500 real images (500 per class), used as the real-domain model-selection/source pool
- `real_test`: 1,500 real images (500 per class), reserved for final evaluation

## Cross-split audit

Exact duplicate detection was performed using SHA-256 file digests. Visual near-duplicate candidates were screened using perceptual hashing (pHash).

### Main held-out test result

No exact SHA-256 duplicate and no flagged pHash near-duplicate candidate was detected between `synthetic_train` and `real_test`.

### Additional audit finding

The audit identified one exact SHA-256 duplicate and six pHash candidate pairs between `real_valid` and `real_test`. These candidates were therefore not ignored. Their influence was assessed using a post-hoc sensitivity analysis based on the already-saved per-image test predictions.

Excluding the single exact duplicate, or conservatively excluding all six flagged test images, produced only negligible changes in the reported model-level accuracy and Macro-F1 values. See `test_overlap_sensitivity_analysis.csv`.

## Files

- `dataset_counts.csv`: class counts by split.
- `synthetic_train_vs_real_test_sha256.csv`: exact duplicate audit for the training/test boundary (0 rows).
- `synthetic_train_vs_real_test_phash.csv`: pHash candidate audit for the training/test boundary (0 rows).
- `real_valid_vs_real_test_sha256.csv`: exact duplicate audit for the model-selection/test boundary.
- `real_valid_vs_real_test_phash.csv`: visual near-duplicate candidates for the model-selection/test boundary.
- `test_overlap_sensitivity_analysis.csv`: recalculated Accuracy and Macro-F1 after excluding flagged test images.
- `real_valid_source_image_ids.csv`: source/model-selection image IDs and hashes.
- `real_test_source_image_ids.csv`: held-out test image IDs and hashes.

## Interpretation

The audit supports the absence of detected exact or flagged near-duplicate overlap between `synthetic_train` and the held-out `real_test` split. Because `real_valid` also served as the source pool for image-to-image synthetic generation, it should be described as a model-selection/source pool rather than as an independent validation set.
