# DV-VLN Data Reuse Status

The R2R annotations, ViT-16 features, dVAE probabilities, and Matterport connectivity prepared for NavCoT are compatible with DV-VLN. The large HDF5 files are linked rather than copied:

- `datasets/R2R/annotations/` links to the NavCoT encoded R2R annotations.
- `datasets/R2R/features/vit-16.hdf5` links to the uploaded ViT features.
- `datasets/R2R/features/dvae_probs.hdf5` links to the uploaded dVAE probabilities.
- `datasets/R2R/connectivity/scans.txt` lists all 90 connectivity scans.

A one-batch `val_unseen_subset` environment smoke test succeeds and returns a `(36, 516)` observation feature tensor. DV-VLN was updated for the installed MatterSim batch API and now tolerates a missing optional caption corpus for non-LLM navigation.

This verifies data and simulator initialization only. Running the trained DV-VLN model still requires its VLN-BERT checkpoint and local BERT configuration; LLaMA assets are only needed for `--llm_predict`.
