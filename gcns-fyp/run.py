import torch
import logging

from optimise import run_experiment
from db import ExperimentDb
from models import SgcMeanPool, SgcStatPool, SgcJumpStatPool, GatMeanPool, GatStatPool, GatJumpStatPool

DB_PATH = "./results/experiments.sqlite"
experiment_db = ExperimentDb(DB_PATH)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

logger = logging.getLogger("run")

for model in (SgcMeanPool, SgcStatPool, SgcJumpStatPool, GatMeanPool, GatStatPool, GatJumpStatPool):
    description = f"HPO for `{model.name}` model on ESOL dataset with GPEI"
    logger.info(f"Running experiment {description}")
    run_experiment(
        device,
        experiment_db,
        description=description,
        num_trials=100,
        model=model,
        dataset_name="esol",
        training_ratio=0.9,
        k_folds=5,
        max_epochs=500,
        metric_name="rmse",
        optimiser_name="adam",
        optimisation_strategy="sobol_gpei"
    )