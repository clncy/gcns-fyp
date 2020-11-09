from typing import Type

from scipy import stats
from torch.utils.data import Subset
from torch_geometric.data import DataLoader, Dataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from sklearn.model_selection import KFold
from pprint import PrettyPrinter

from models import ModelBase

pp = PrettyPrinter()

def trial_factory(model_cls: Type[ModelBase], train_dataset: Dataset, k_folds: int, max_epochs: int):
    """ Creates a closure around the trial fucntion to give access to additional parameters

    Given that the trial function can only accept a single `params` argument, this
    factory function is used to allow the specification of additional parameters,
    rather than requiring them to be pulled in from the global scope.

    """

    def trial(params: dict):
        """ Completes a full trial of training a model with given hyperparameters

        This function performs k-fold validation on a model with the given set of
        hyperparameters in order to estimate the performance of the model on the test
        set. This estimate is returned to the hyperparameter optimiser so that it can
        decide on the best set of hyperparameters for the model

        """

        print("Running trial with following parameters:")
        print(params)

        trainer = pl.Trainer(
            callbacks=[EarlyStopping(monitor="val_loss")],
            max_epochs=max_epochs
        )
        validation_losses = []
        kf = KFold(n_splits=k_folds)
        for train_index, val_index in kf.split(range(len(train_dataset))):
            train_loader = DataLoader(
                Subset(train_dataset, train_index), 
                batch_size=params["batch_size"]
            )
            val_loader = DataLoader(
                Subset(train_dataset, val_index), 
                shuffle=False
            )

            # Create a model for each set of cross-validation, with the given hyperparameters
            model = model_cls(
                hidden_channels=2 ** params["channels_power_of_two"],
                representation_size=params["representation_size"],
                output_size=params["output_size"],
                embedding_dimension=params["embedding_dimension"],
                hyper_params=params,
            )

            # Fit the model to the given folds of the training data
            trainer.fit(model, train_loader, val_loader)

            # Find the (now biased) validation loss
            validation_losses.append(trainer.logged_metrics["val_loss"])

        mean_loss = stats.tmean(validation_losses)
        sem_loss = stats.sem(validation_losses) if len(validation_losses) > 1 else 0.0

        # Log the results of the trial
        print(f"Optimisation trial complete: Mean validation loss {mean_loss}, SEM {sem_loss}")
        print("The following hyperparameters were used:")
        pp.pprint(params)

        return {"validation_loss": (mean_loss, sem_loss)}
    return trial