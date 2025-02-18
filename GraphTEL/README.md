
# My Paper Title

This repository is the official implementation of [Structure-aware Self-supervised Graph Representation Learning]().

## Requirements

To install requirements:

```setup
pip install -r requirements.txt
```

## Training

To train the model(s) in the paper, run this command:

```train
python m_graph.py --dataset "IMDB-B" --seeds [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] --device "cuda:14" --max_epoch 200 --num_layers 2 --num_hidden 256 --residual --lr 0.005 --use5 True --use6 True --weight_decay 5e-4 --drop_edge_rate 0.5 --loss_fn sce --alpha_l 2 --optimizer adam --max_epoch_f 30 --lr_f 0.001 --weight_decay_f 0.0 --linear_prob True --scheduler False --concat_hidden False --deg4feat False --batch_size 32

## Evaluation

It will automatically evaluate the model in the same py file after completing the training process.

## Results

Our model achieves the following performance on IMDB-BINARY:


| Model name         | Accuracy  |
| ------------------ |---------------- |
| GraphTEL   |     77.40±0.31         |
