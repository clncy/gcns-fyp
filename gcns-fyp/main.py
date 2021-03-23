from logging import Logger
import torch
from ax import optimize
from torch.nn import MSELoss
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet

from models import GCN
from optimise import trial_factory

logger = Logger("main")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    logger.info("CUDA is available")
else:
    logger.info("No CUDA available")

logger.info(device)

TRAINING_RATIO = 0.8 # The ratio of the dataset to be utilised for training/validation

dataset = MoleculeNet(root="data/Lipo", name="Lipo")
dataset = dataset.shuffle()

train_split = round(TRAINING_RATIO * len(dataset))
test_split = len(dataset) - train_split
train_dataset, test_dataset = random_split(dataset, lengths=[train_split, test_split])

# Construct the trial function that will evaluate a set of hyperparameters
trial = trial_factory(
    model_cls=GCN, 
    train_dataset=train_dataset, 
    k_folds=4, 
    max_epochs=100
)

# Define the space of hyperparameters
parameters = [
        dict(
            name="lr",
            bounds=[1e-6, 1.0],
            type="range",
            log_scale=True,
            value_type="float",
        ),
        dict(
            name="channels_power_of_two", bounds=[2, 3], value_type="int", type="range"
        ),
        dict(name="representation_size", value=64, value_type="int", type="fixed"),
        dict(name="output_size", value=1, value_type="int", type="fixed"),
        dict(name="embedding_dimension", value=16, value_type="int", type="fixed"),
        dict(name="batch_size", value=64, value_type="int", type="fixed"),
    ]

# Perform hyperparameter optimisation
results = optimize(
    parameters=parameters,
    evaluation_function=trial,
    minimize=True,
    objective_name="validation_loss",
    total_trials=2,
)
