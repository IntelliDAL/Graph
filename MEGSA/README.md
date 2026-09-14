# MEGSA

PyTorch implementation of **MEGSA: Adaptive Motif-Level Structural Alignment for Graph Similarity Learning**. MEGSA stands for *Motif-Enhanced Graph Similarity Alignment*.

The model uses two stages: self-supervised motif extraction through decomposition and reconstruction, followed by motif interaction graph similarity learning. See the [source module guide](src/README.md).

## Installation

Use **Python 3.10**. The tested local environment uses Python 3.10.14, PyTorch 2.1.1, and PyTorch Geometric 2.6.1. Dependencies are pinned in [requirements.txt](requirements.txt).

Create and activate an environment:

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install PyTorch for your platform, then the remaining dependencies. For a CPU build on Windows or Linux:

```bash
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
```

For CUDA 12.1 on Windows or Linux, replace the first installation command with:

```bash
python -m pip install torch==2.1.1 --index-url https://download.pytorch.org/whl/cu121
```

On macOS, install `torch==2.1.1` from the default package index, then install `requirements.txt`; this project selects CPU on systems without CUDA. These version-specific build commands follow the [official PyTorch installation archive](https://pytorch.org/get-started/previous-versions/#v211). Only the local Windows environment has been tested here.

## Repository layout

```text
MEGSA/
  README.md
  requirements.txt
  src/                 # Model, data pipeline, CLI, and training code
  datasets/            # Downloaded datasets (created locally, Git-ignored)
  pretrain_model/      # Motif extractor checkpoints (Git-ignored)
  model/               # Downstream checkpoints (Git-ignored)
```

Default storage paths are anchored to the `MEGSA/` project directory, regardless of the working directory. Override them with `--data_dir`, `--pretrain_dir`, and `--model_dir`. Explicit relative overrides are resolved against the working directory.

## Data

Supported datasets: `AIDS700nef`, `LINUX`, and `IMDBMulti`. `GEDDataset` downloads and processes missing data under `datasets/<dataset>/`; internet access is required on first use. Existing processed data is reused. Dataset downloads depend on upstream hosting and are not bundled in the repository.

## Training and evaluation

Run these commands from the `MEGSA/` project directory (run `cd MEGSA` first when starting from the Graph repository root). `python src/main.py` is also supported. To run from another directory, use the absolute path to `src/main.py`.

```bash
python -m src.main --help
```

Train both stages and evaluate:

```bash
python -m src.main --mode all --dataset AIDS700nef --device cpu
```

Or run each stage separately:

```bash
python -m src.main --mode pretrain --dataset AIDS700nef --device cpu
python -m src.main --mode train --dataset AIDS700nef --device cpu
python -m src.main --mode eval --dataset AIDS700nef --device cpu
```

Use `--device 0` or `--device cuda:0` for the first GPU. When CUDA is unavailable, execution falls back to CPU. Checkpoints are loaded onto the selected device.

`pretrain` saves extractor weights; `train` loads extractor weights and saves downstream weights; `eval` loads both checkpoints; `all` runs all three stages. The default mode is `eval`, which requires existing checkpoints. Training saves to the configured checkpoint names and replaces existing files at those paths. Use separate directory overrides for separate experiments.

Checkpoint names remain compatible with existing state dictionaries:

```text
pretrain_model/pretrain_model_AIDS700nef_5.pth
model/model_AIDS700nef_5.pth
```

Weights are not bundled by default, and no public download URL is currently provided. Train with `--mode all`, or place compatible weights at the paths above. Missing weights are reported before datasets are loaded. Library callers should call `Trainer.load()` before evaluating saved weights with `Trainer.score()`.

For a short workflow check, use one epoch per training stage and separate output directories:

```bash
python -m src.main --mode all --dataset AIDS700nef --device cpu --pretrain_epochs 1 --epochs 1 --pretrain_dir outputs/smoke/pretrain --model_dir outputs/smoke/model
```

This still evaluates the full test set and does not reproduce the paper's reported results. Evaluation compares each test graph against the training gallery; it currently processes the entire gallery at once, so `--batch_size` controls training rather than evaluation memory usage.

## Default configuration

| Parameter | Default |
| --- | --- |
| Batch size | 128 |
| Pretraining epochs | 3000 |
| Downstream epochs | 30000 |
| Learning rate | 0.0001 |
| Weight decay | 0.0001 |
| Dropout | 0.2 |
| Motif interaction rounds | 2 |

| Dataset | Node capacity | Feature dimension | Motifs | Structural loss weight |
| --- | --- | --- | --- | --- |
| AIDS700nef | 10 | 29 | 5 | 30 |
| LINUX | 10 | 8 | 5 | 100 |
| IMDBMulti | 89 | 89 | 10 | 200 |

The release setup preserves the existing network, losses, dataset preprocessing, and parameter defaults. It does not add validation-based hyperparameter search or alter the experimental protocol. Successful execution does not establish reproduction of the paper's numerical results.

## Paper reference

Zhuolin Jia, Guangqi Wen, Lingwen Liu, Peng Cao, Jinzhu Yang, and Osmar R. Zaiane. **MEGSA: Adaptive Motif-Level Structural Alignment for Graph Similarity Learning.** Publication venue, DOI, and year are not specified here pending confirmed publication metadata.

## License

A project license has not yet been specified. No license grant is supplied by this repository. Dataset and third-party package terms remain separate.
