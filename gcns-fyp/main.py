import os
import time
import logging

import torch
from csv import DictWriter
from ax import optimize, Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep
from torch.nn import MSELoss
from torch.utils.data import random_split
from torch_geometric.datasets import MoleculeNet

# Set logging formatting
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)

from models import GAT, MultiGCN, ConcatPool, ResConcatPool
from optimise import TrialRunner


TRAINING_RATIO = 0.9 # The ratio of the dataset to be utilised for training/validation
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger = logging.getLogger("autograph")
logger.info(f"Device type: {device.type}")


dataset_generators = dict(
    lipo=lambda: MoleculeNet(root="data/Lipo", name="Lipo"),
    esol=lambda: MoleculeNet(root="data/ESOL", name="ESOL")
)
dataset = dataset_generators["esol"]()
dataset = dataset.shuffle()

train_split = round(TRAINING_RATIO * len(dataset))
test_split = len(dataset) - train_split
train_dataset, test_dataset = random_split(dataset, lengths=[train_split, test_split])

mse_loss = MSELoss()
rmse_loss = lambda y_hat, y: torch.sqrt(mse_loss(y_hat, y))

# Construct the trial function that will evaluate a set of hyperparameters
trial_runner = TrialRunner(
    model_cls=ResConcatPool,
    train_dataset=train_dataset,
    test_dataset=test_dataset,
    k_folds=5,
    max_epochs=500,
    criterion=rmse_loss,
    optimiser_cls=torch.optim.Adam,
    device=device
)

# Define the space of hyperparameters
old_parameters = [
        dict(
            name="lr",
            bounds=[1e-6, 1e-2],
            type="range",
            log_scale=True,
            value_type="float",
        ),
        dict(name="batch_size", values=[64, 128, 256], value_type="int", type="choice"),
        dict(name="hidden_channels_power_of_two", bounds=[5, 8], value_type="int", type="range"),
        dict(name="dropout_probability", bounds=[0.3, 0.9], value_type="float", type="range"),
        dict(name="representation_size", values=[32, 64, 128], value_type="int", type="choice"),
        dict(name="hidden_layers", bounds=[0, 4], value_type="int", type="range"),
        dict(name="output_size", value=1, value_type="int", type="fixed"),
        dict(name="embedding_dimension", value=100, value_type="int", type="fixed"),
]

parameters = [
        dict(
            name="lr",
            bounds=[1e-6, 1e-2],
            type="range",
            log_scale=True,
            value_type="float",
        ),
        dict(name="batch_size", bounds=[16, 256], value_type="int", type="range"),
        dict(name="hidden_channels_power_of_two", bounds=[5, 8], value_type="int", type="range"),
        dict(name="dropout_probability", bounds=[0.3, 0.9], value_type="float", type="range"),
        dict(name="representation_size", bounds=[32, 128], value_type="int", type="range"),
        dict(name="hidden_layers", bounds=[0, 4], value_type="int", type="range"),
        dict(name="output_size", value=1, value_type="int", type="fixed"),
        dict(name="embedding_dimension", value=100, value_type="int", type="fixed"),
]

# Perform hyperparameter optimisation
NUM_TRIALS = 1
logger.info(f"Running experiment for {NUM_TRIALS} trial")

gs = GenerationStrategy(
    steps=[
        GenerationStep(
            model=Models.SOBOL,
            num_trials=NUM_TRIALS
        ),
        # GenerationStep(
        #     model=Models.GPEI,
        #     num_trials=int(0.8*NUM_TRIALS)
        # )
    ]
)
best_parameters, values, experiment, model = optimize(
    parameters=parameters,
    evaluation_function=trial_runner.run,
    minimize=True,
    objective_name="validation_loss",
    total_trials=NUM_TRIALS,
    generation_strategy=gs
)

# Export the results
epoch_time = int(time.time())
output_dir = "./results"
file_name = f"experiment_{epoch_time}.csv"
output_path = os.path.join(output_dir, file_name)

logger.info(f"Completed experiment, outputting results to {output_path}")
with open(output_path, mode="w") as fp:
    fieldnames = list(trial_runner.results[0].keys())
    dict_writer = DictWriter(fp, fieldnames=fieldnames)

    dict_writer.writeheader()
    dict_writer.writerows(trial_runner.results)
