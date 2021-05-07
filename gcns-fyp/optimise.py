import time
import torch
import logging
from scipy import stats
from torch.utils.data import Subset
from torch_geometric.data import DataLoader
from sklearn.model_selection import KFold
from pprint import PrettyPrinter
pp = PrettyPrinter()


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
    def __init__(self, model_cls, train_dataset, test_dataset, k_folds, max_epochs, criterion, optimiser_cls, device):
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

        for epoch in range(1, self.max_epochs+1):
            epoch_losses = []
            for data in train_loader:
                data = data.to(self.device)
                out = model(data.x, data.edge_index, data.batch)
                loss = self.criterion(out, data.y)
                epoch_losses.append(loss.item())
                loss.backward()
                optimiser.step()
                optimiser.zero_grad()

            mean_epoch_loss = sum(epoch_losses)/len(epoch_losses)
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

    def run_trial(self, hidden_channels_power_of_two, representation_size, output_size, embedding_dimension, hidden_layers, dropout_probability, lr, batch_size):
        train_losses = []
        validation_losses = []
        test_losses = []
        kf = KFold(n_splits=self.k_folds)

        self.logger.info(f"Running optimisation trial {self.trial_number}")
        start_time = time.time()
        for fold_number, (train_index, val_index) in enumerate(kf.split(range(len(self.train_dataset)))):
            train_subset = Subset(self.train_dataset, train_index)
            val_subset = Subset(self.train_dataset, val_index)
            train_loader = DataLoader(
                train_subset,
                batch_size=batch_size
            )

            # Create a model for each set of cross-validation, with the given hyperparameters
            model = self.model_cls(
                embedding_dimension,
                2 ** hidden_channels_power_of_two,
                hidden_layers,
                representation_size,
                output_size,
                dropout_probability
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

            self.logger.info(f"Completed cross-validation fold {fold_number+1} after {epochs_completed} epochs")

        train_loss, train_sem = self.compute_loss_statistics(train_losses)
        val_loss, val_sem = self.compute_loss_statistics(validation_losses)
        test_loss, test_sem = self.compute_loss_statistics(test_losses)

        end_time = time.time()
        self.logger.info(f"Completed trial in {end_time - start_time}s")
        results = dict(
            trial_number=self.trial_number,
            hidden_layers=hidden_layers,
            dropout_probability=dropout_probability,
            channels_power_of_two=hidden_channels_power_of_two,
            representation_size=representation_size,
            output_size=output_size,
            embedding_dimension=embedding_dimension,
            lr=lr,
            batch_size=batch_size,
            train_loss=train_loss,
            train_sem=train_sem,
            val_loss=val_loss,
            val_sem=val_sem,
            test_loss=test_loss,
            test_sem=test_sem
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