# DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Research%20Use-lightgrey)](#license)
[![GitHub](https://img.shields.io/badge/Code-DV--VLN-181717?logo=github)](https://github.com/xuenanhkust163/DV-VLN)

Official PyTorch implementation of **DV-VLN: Dual Verification for Reliable
LLM-Based Vision-and-Language Navigation**.

DV-VLN uses a generate-then-verify pipeline for language-mediated navigation.
At each navigation step, the LLM samples candidate actions in a structured
Prediction-View Match-Action format. True-False Verification (TFV) checks
global consistency, while Masked-Entity Verification (MEV) measures
fine-grained landmark alignment. Their normalized scores rerank the candidate
actions before execution.

## Highlights

- Dual verification with complementary TFV and MEV signals.
- Inference-time reranking without training a separate verifier.
- Evaluation support for R2R, R4R, and the English subset of RxR.
- Ablation modes for direct prediction, majority vote, TFV, MEV, and DV-VLN.
- Ranking diagnostics, MEV controls, bounded smoke tests, and GPU-memory logs.

## Method at a Glance

```text
instruction + history + candidate views
                    |
                    v
       sample K navigation candidates
                    |
          +---------+---------+
          |                   |
          v                   v
    TFV consistency      MEV entity recovery
          |                   |
          +---------+---------+
                    |
                    v
        normalized score fusion
                    |
                    v
          selected navigation action
```

## Installation

Clone the repository and create a Python environment:

```bash
git clone https://github.com/xuenanhkust163/DV-VLN.git
cd DV-VLN

conda create -n dvvln python=3.10 -y
conda activate dvvln
pip install -r requirements.txt
pip install -r LLaMA2-Accessory/requirements.txt
```

Build Matterport3D Simulator following its
[upstream instructions](third_party/Matterport3DSimulator/README.md). The
included CMake option supports a system installation of modern pybind11:

```bash
cd third_party/Matterport3DSimulator
mkdir -p build && cd build
cmake .. -DUSE_SYSTEM_PYBIND11=ON
make -j
```

Add the resulting Python module to `PYTHONPATH` when it is not installed into
the active environment.

## Data Preparation

Download the navigation annotations, connectivity graphs, pre-extracted
CLIP-ViT/16 features, dVAE probabilities, and candidate captions. The assets
used by NavCoT are compatible with this implementation; see the
[NavCoT data instructions](https://github.com/expectorlin/NavCoT#data-preparation)
for the original download sources.

Arrange them as follows. Large datasets and model weights are intentionally not
stored in Git.

```text
datasets/
└── R2R/
    ├── annotations/
    │   ├── R2R_train_enc.json
    │   ├── R2R_val_seen_enc.json
    │   ├── R2R_val_unseen_enc.json
    │   └── R2R_val_unseen_subset_enc.json
    ├── captions/
    │   └── <scan>/<viewpoint>/<scan>_<viewpoint>.json
    ├── connectivity/
    │   ├── scans.txt
    │   └── *_connectivity.json
    └── features/
        ├── vit-16.hdf5
        └── dvae_probs.hdf5
```

RxR annotations belong under `datasets/RxR/annotations/`. If the data lives
elsewhere, set `DVVLN_DATA_ROOT` instead of creating machine-specific links in
the repository.

Check the local assets before running an experiment:

```bash
./check_runtime_assets.sh
```

## Model Preparation

LLM navigation requires:

- a LLaMA2-Accessory compatible checkpoint directory;
- its `tokenizer.model`;
- candidate captions for the selected benchmark;
- an optional independent Hugging Face verifier for verifier-control studies.

The repository does not redistribute model weights. Supply local paths through
environment variables:

```bash
export DVVLN_PRETRAINED_PATH=/data/models/dv-vln-llama
export DVVLN_TOKENIZER_PATH=/data/models/llama/tokenizer.model
export DVVLN_DATA_ROOT=/data/dv-vln/datasets
export DVVLN_CAPTION_DIR=/data/dv-vln/datasets/R2R/captions
```

No source file needs to be edited for a different machine.

## Evaluation

Run from the repository root or any other directory. The scripts resolve the
repository path automatically.

```bash
bash finetune_src/scripts/run_r2r.sh
bash finetune_src/scripts/run_r4r.sh
bash finetune_src/scripts/run_rxr.sh
```

Useful optional settings:

```bash
export DVVLN_GPU=0
export DVVLN_SEED=0
export DVVLN_OUTPUT_DIR=/data/experiments/dv-vln/r2r
export DVVLN_VERIFICATION_MODE=dual
export DVVLN_NUM_SAMPLES=4
export DVVLN_VERIFICATION_ATTEMPTS=4
export DVVLN_TOP_P=0.9
```

For a short pipeline check, add arguments after the script name:

```bash
bash finetune_src/scripts/run_r2r.sh \
  --eval_splits val_unseen_subset \
  --eval_max_instrs 10
```

The primary verification modes are:

| Mode | Candidate selection |
| --- | --- |
| `direct` | Direct LLM action prediction |
| `vote` | Majority vote over sampled candidates |
| `tfv` | True-False Verification only |
| `mev` | Masked-Entity Verification only |
| `dual` | Combined TFV and MEV (DV-VLN) |

Additional diagnostic modes are listed by:

```bash
cd finetune_src
python r2r/main.py --help
```

## Reproducibility Notes

- The manuscript protocol uses `K=4`, `top_p=0.9`, four verification attempts,
  and a maximum R2R action length of 15.
- Keep the dataset split, random seed, candidate budget, and verification
  budget fixed when comparing ablations.
- `--eval_max_instrs` is intended only for smoke tests, not reported results.
- Record the Git commit, checkpoint identifiers, environment, and output logs
  for every formal run.
- Data and checkpoints are external assets. A successful clone alone is not a
  complete reproduction until `check_runtime_assets.sh` reports the required
  files.

## Repository Structure

```text
DV-VLN/
├── finetune_src/                    # Navigation environment and DV-VLN agent
│   ├── r2r/                         # Data, evaluation, and verification logic
│   ├── models/                      # VLN backbone
│   ├── scripts/                     # R2R, R4R, and RxR launchers
│   └── utils/                       # Prompts, logging, and distributed helpers
├── LLaMA2-Accessory/                # LLM backbone integration
├── third_party/Matterport3DSimulator/
├── datasets/                        # Local data mount point (large files ignored)
└── check_runtime_assets.sh          # Preflight asset check
```

## Citation

If this repository is useful in your research, please cite:

```bibtex
@article{xue2026dvvln,
  title   = {DV-VLN: Dual Verification for Reliable LLM-Based Vision-and-Language Navigation},
  author  = {Xue, Nan and Zhang, Meimei and Li, Zijun and Li, Shijie and Zhang, Zhenxi and Li, Bin and Zhou, Shoujun},
  year    = {2026}
}
```

## Acknowledgements

This codebase builds on
[NavCoT](https://github.com/expectorlin/NavCoT),
[VLN-HAMT](https://github.com/cshizhe/VLN-HAMT),
[VLN-SIG](https://github.com/jialuli-luka/VLN-SIG),
[LLaMA2-Accessory](https://github.com/Alpha-VLLM/LLaMA2-Accessory), and
[Matterport3D Simulator](https://github.com/peteanderson80/Matterport3DSimulator).
We thank their authors for releasing their implementations.

## License

Please follow the licenses of this repository and its third-party components.
Matterport3D data and LLaMA model weights are subject to their respective terms.
