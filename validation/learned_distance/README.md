# Learned distance for the replicon classifier

Validation of the optional learned distance (Supplementary Note 3).

A supervised contrastive encoder maps the 256-dimensional tetranucleotide vectors into an embedding; the replicon classifier is unchanged (k = 5, cosine, distance-weighted). Encoders are trained within each fold of stratified 5-fold cross-validation on training plasmids only.

Data: `data/inc_classifier.npz` (8,404 plasmids, 29 groups).

| | Accuracy | Macro F1 | Weighted F1 | Balanced accuracy |
|---|---|---|---|---|
| Raw tetranucleotide distance | 0.917 | 0.687 | 0.912 | 0.677 |
| Contrastive encoder, random batches (`validate_encoder_vs_knn.py`) | 0.928 | 0.693 | 0.923 | 0.681 |
| Contrastive encoder, class-balanced batches (`run_encoder.py`) | 0.933 | 0.735 | 0.930 | 0.717 |

The class-balanced encoder scores highest on macro F1 in all five folds (`results_head2head.json`).

## Run

From the repository root:

    python validation/learned_distance/head2head.py

## Files

- `run_encoder.py`: class-balanced encoder (from https://github.com/gijsdanoe/plin-learned-distance)
- `head2head.py`: raw kNN, `validate_encoder_vs_knn.py` encoder and class-balanced encoder on the same folds
- `results_head2head.json`: overall and per-fold results
- `results_default.json`: per-group F1 and neighbour purity (Supplementary Note 3, Supplementary Table 1)
