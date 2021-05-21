import time
import torch
import logging
from scipy import stats
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from pprint import PrettyPrinter
from torch_geometric.datasets import MoleculeNet
from torch.nn import MSELoss
from torch.utils.data import random_split
from ax import optimize, Models
from ax.modelbridge.generation_strategy import GenerationStrategy, GenerationStep

# Set logging formatting. Must be done before model import
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s"
)

pp = PrettyPrinter()

logger = logging.getLogger("optimise")

class EarlyStopping:
    def __init__(self, patience=5, eps=1e-4):
        self.patience = patience
        self.eps = eps
        self.previous_loss = None
        self.epochs_without_improvement = 0

    def step(self, loss):
        if self.previous_loss:
            improvement = (self.previous_loss - loss) > self.eps
            if improvement:
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1

        self.previous_loss = loss

    @property
    def should_early_stop(self):
        return self.epochs_without_improvement >= self.patience


class TrialRunner:
    def __init__(
        self,
        experiment_db,
        experiment_id,
        model_cls,
        train_dataset,
        test_dataset,
        k_folds,
        max_epochs,
        criterion,
        optimiser_cls,
        device,
    ):
        self.experiment_db = experiment_db
        self.experiment_id = experiment_id
        self.model_cls = model_cls
        self.k_folds = k_folds
        self.max_epochs = max_epochs
        self.criterion = criterion
        self.optimiser_cls = optimiser_cls
        self.device = device
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.trial_number = 0
        self.results = list()
        self.logger = logging.getLogger("autograph").getChild("trial")

    def train_model(self, model, train_loader, optimiser):
        model.train()
        es = EarlyStopping()

        for epoch in range(1, self.max_epochs + 1):
            epoch_losses = []
            for data in train_loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(out, data.y)
                epoch_losses.append(loss.item())
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            mean_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            # print(f"Epoch {epoch} completed with average loss {mean_epoch_loss}")

            es.step(mean_epoch_loss)
            if es.should_early_stop:
                # print(f"Early stopping at epoch {epoch}")
                return epoch

        return self.max_epochs

    def evaluate_model(self, model, dataset):
        model.eval()
        data_loader = DataLoader(dataset, batch_size=len(dataset))
        data = next(iter(data_loader))
        data = data.to(self.device)

        with torch.no_grad():
            out = model(data.x, data.edge_index, data.batch)
            loss = self.criterion(out, data.y)

        return loss.item()

    def compute_loss_statistics(self, losses):
        mean_loss = stats.tmean(losses)
        sem_loss = stats.sem(losses)

        return mean_loss, sem_loss

    def run_trial(
        self,
        hidden_channels_power_of_two,
        representation_size,
        output_size,
        embedding_dimensions,
        hidden_layers,
        dropout_probability,
        lr,
        batch_size,
    ):
        train_losses = []
        validation_losses = []
        test_losses = []
        kf = KFold(n_splits=self.k_folds)

        hidden_channels = 2 ** hidden_channels_power_of_two

        self.logger.info(f"Running optimisation trial {self.trial_number}")
        trial_id = self.experiment_db.create_trial(
            experiment_id=self.experiment_id,
            trial_number=self.trial_number,
            learning_rate=lr,
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            dropout_probability=dropout_probability,
            representation_size=representation_size,
            hidden_layers=hidden_layers,
            embedding_dimensions=embedding_dimensions,
        )
        trial_start_time = time.time()
        for fold_index, (train_index, val_index) in enumerate(
            kf.split(range(len(self.train_dataset)))
        ):
            fold_number = fold_index + 1
            fold_start_time = time.time()
            train_subset = Subset(self.train_dataset, train_index)
            val_subset = Subset(self.train_dataset, val_index)
            train_loader = DataLoader(train_subset, batch_size=batch_size)

            # Create a model for each set of cross-validation, with the given hyperparameters
            model = self.model_cls(
                embedding_dimensions,
                hidden_channels,
                hidden_layers,
                representation_size,
                output_size,
                dropout_probability,
            ).to(self.device)

            optimiser = self.optimiser_cls(model.parameters(), lr=lr)
            # Fit the model to the given folds of the training data
            epochs_completed = self.train_model(model, train_loader, optimiser)

            # Evaluate the model performance on this fold of the data
            train_loss = self.evaluate_model(model, train_subset)
            train_losses.append(train_loss)

            val_loss = self.evaluate_model(model, val_subset)
            validation_losses.append(val_loss)

            test_loss = self.evaluate_model(model, self.test_dataset)
            test_losses.append(test_loss)

            fold_end_time = time.time()
            self.logger.info(
                f"Completed cross-validation fold {fold_number} after {epochs_completed} epochs"
            )

            self.experiment_db.create_fold(
                trial_id=trial_id,
                fold_number=fold_number,
                duration_secs=fold_end_time-fold_start_time,
                train_loss=train_loss,
                val_loss=val_loss,
                test_loss=test_loss,
                epochs=epochs_completed
            )

        train_loss, train_sem = self.compute_loss_statistics(train_losses)
        val_loss, val_sem = self.compute_loss_statistics(validation_losses)
        test_loss, test_sem = self.compute_loss_statistics(test_losses)

        end_time = time.time()
        self.logger.info(f"Completed trial in {end_time - trial_start_time}s")
        results = dict(
            trial_number=self.trial_number,
            hidden_layers=hidden_layers,
            dropout_probability=dropout_probability,
            channels_power_of_two=hidden_channels_power_of_two,
            representation_size=representation_size,
            output_size=output_size,
            embedding_dimensions=embedding_dimensions,
            lr=lr,
            batch_size=batch_size,
            train_loss=train_loss,
            train_sem=train_sem,
            val_loss=val_loss,
            val_sem=val_sem,
            test_loss=test_loss,
            test_sem=test_sem,
        )
        self.results.append(results)

        # Log the results of the trial
        pp.pprint(results)

        return {
            "validation_loss": (val_loss, val_sem),
        }

    def run(self, params: dict):
        self.trial_number += 1
        return self.run_trial(**params)


def run_experiment(
    device,
    experiment_db,
    description,
    num_trials,
    model,
    dataset_name,
    training_ratio,
    k_folds,
    max_epochs,
    metric_name,
    optimiser_name,
    optimisation_strategy,
):

    dataset_generators = dict(
        lipo=lambda: MoleculeNet(root="data/Lipo", name="Lipo"),
        esol=lambda: MoleculeNet(root="data/ESOL", name="ESOL"),
    )

    mse_loss = MSELoss()
    metric_lookup = dict(
        mse=mse_loss,
        rmse=lambda y_hat, y: torch.sqrt(mse_loss(y_hat, y)),
    )

    optimiser_lookup = dict(adam=torch.optim.Adam)

    optimisation_strategies = dict(
        sobol=GenerationStrategy(
            steps=[GenerationStep(model=Models.SOBOL, num_trials=num_trials)]
        ),
        sobol_gpei=GenerationStrategy(
            steps=[
                GenerationStep(model=Models.SOBOL, num_trials=0.25*num_trials),
                GenerationStep(model=Models.GPEI, num_trials=0.75*num_trials)
            ]
        )
    )

    # Generate the train/test split
    dataset = dataset_generators[dataset_name]()
    dataset = dataset.shuffle()

    train_split = round(training_ratio * len(dataset))
    test_split = len(dataset) - train_split
    train_dataset, test_dataset = random_split(
        dataset, lengths=[train_split, test_split]
    )

    # Create an entry in the DB for the experiment
    experiment_id = experiment_db.create_experiment(
        description=description,
        model=model.name,
        dataset=dataset_name,
        trials=num_trials,
        training_split=training_ratio,
        optimisation_strategy=optimisation_strategy,
        max_epochs=max_epochs,
        optimiser=optimiser_name,
        metric=metric_name,
    )

    trial_runner = TrialRunner(
        experiment_db=experiment_db,
        experiment_id=experiment_id,
        model_cls=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        k_folds=k_folds,
        max_epochs=max_epochs,
        criterion=metric_lookup[metric_name],
        optimiser_cls=optimiser_lookup[optimiser_name],
        device=device,
    )

    parameters = [
        dict(
            name="lr",
            bounds=[1e-6, 1e-2],
            type="range",
            log_scale=True,
            value_type="float",
        ),
        dict(name="batch_size", bounds=[16, 256], value_type="int", type="range"),
        dict(
            name="hidden_channels_power_of_two",
            bounds=[5, 8],
            value_type="int",
            type="range",
        ),
        dict(
            name="dropout_probability",
            bounds=[0.3, 0.9],
            value_type="float",
            type="range",
        ),
        dict(
            name="representation_size", bounds=[32, 128], value_type="int", type="range"
        ),
        dict(name="hidden_layers", bounds=[0, 4], value_type="int", type="range"),
        dict(name="output_size", value=1, value_type="int", type="fixed"),
        dict(name="embedding_dimensions", value=100, value_type="int", type="fixed"),
    ]

    logger.info(f"Running experiment for {num_trials} trial")

    best_parameters, values, experiment, model = optimize(
        parameters=parameters,
        evaluation_function=trial_runner.run,
        minimize=True,
        objective_name="validation_loss",
        total_trials=num_trials,
        generation_strategy=optimisation_strategies[optimisation_strategy],
    )

    logger.info(f"Best parameters", best_parameters)