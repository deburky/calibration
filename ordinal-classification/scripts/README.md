# Scripts

This directory contains scripts for training and evaluating ordinal neural networks.

The following scripts are available:

- `ordinal_training_2_layers.py`: Train an ordinal neural network with 2 hidden layers.
- `ordinal_training_3_layers.py`: Train an ordinal neural network with 3 hidden layers.
- `ordinal_training_transformer.py`: Train an ordinal transformer.

For benchmarking, we train standard multi-class networks:

- `softmax_training_2_layers.py`: Train a multi-class network with 2 hidden layers.
- `softmax_training_3_layers.py`: Train a multi-class network with 3 hidden layers.
- `softmax_training_transformer.py`: Train a multi-class transformer.

We use Optuna to optimize hyperparameters.

Run the scripts as follows:
    
```bash
python3 ordinal_training_2_layers.py
```
