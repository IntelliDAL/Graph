# Source modules

See the [project README](../README.md) for installation, datasets, commands, checkpoints, and licensing status.

| File | Responsibility |
| --- | --- |
| `main.py` | CLI dispatch for pretrain, train, eval, and all |
| `args.py` | Runtime arguments and dataset configuration |
| `runtime.py` | Repository paths and CPU/CUDA device selection |
| `data.py` | GED datasets and dense graph batches |
| `model.py` | SelfSupervisedMotifExtraction and MEGSA |
| `layers.py` | NodeEncoding, MotifAssignment, GNNEncoder, MotifDecoding, MotifLevelAlignment, GraphLevelMatching, and SimilarityPrediction |
| `train.py` | Training, checkpoint I/O, and evaluation |
| `utils.py` | Metric reporting and precision at k |

In Figure 3 of the paper, motif decomposition corresponds to node encoding followed by motif assignment. Motif reconstruction corresponds to motif encoding, assignment-weighted fusion, and motif decoding. Parameter attribute names are retained for compatibility with existing state dictionaries. Code comments and docstrings are in English.
