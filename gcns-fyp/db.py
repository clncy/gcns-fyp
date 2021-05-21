"""
Wrapper class that saves data to the experiment DB
"""
import os
import datetime
import pathlib
import sqlite3

SCHEMA_PATH = os.path.join(pathlib.Path().absolute(), "db_schema.sql")


class ExperimentDb:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(db_path)
        self.init_db()

    def init_db(self):
        with open(SCHEMA_PATH) as fp:
            init_script = fp.read()

            with self.conn:
                self.conn.executescript(init_script)

    @staticmethod
    def current_datetime():
        return datetime.datetime.now().isoformat()

    def insert_row(self, table, data):
        params, values = zip(*data.items())
        column_string = f"({','.join(params)})"
        params_string = f"({','.join(len(params) * ['?'])})"
        with self.conn:
            cursor = self.conn.execute(
                f"INSERT INTO {table}{column_string} VALUES {params_string} RETURNING id", values
            )

            id = cursor.fetchone()[0]

        return id

    def create_experiment(
        self,
        description: str,
        model: str,
        dataset: str,
        trials: int,
        training_split: float,
        optimisation_strategy: str,
        max_epochs: int,
        optimiser: str,
        metric: str,
    ):

        experiment_datetime = self.current_datetime()
        data = dict(
            description=description,
            datetime=experiment_datetime,
            model=model,
            dataset=dataset,
            trials=trials,
            training_split=training_split,
            optimisation_strategy=optimisation_strategy,
            max_epochs=max_epochs,
            optimiser=optimiser,
            metric=metric,
        )
        return self.insert_row("experiment", data)

    def create_trial(
        self,
        experiment_id: int,
        trial_number: int,
        learning_rate: float,
        batch_size: int,
        hidden_channels: int,
        dropout_probability: float,
        representation_size: int,
        hidden_layers: int,
        embedding_dimensions: int,
    ):
        data = dict(
            experiment_id=experiment_id,
            trial_number=trial_number,
            learning_rate=learning_rate,
            batch_size=batch_size,
            hidden_channels=hidden_channels,
            dropout_probability=dropout_probability,
            representation_size=representation_size,
            hidden_layers=hidden_layers,
            embedding_dimensions=embedding_dimensions
        )

        return self.insert_row("trial", data)

    def create_fold(
        self,
        trial_id: int,
        fold_number: int,
        duration_secs: float,
        train_loss: float,
        val_loss: float,
        test_loss: float,
        epochs: int,
    ):

        data = dict(
            trial_id=trial_id,
            fold_number=fold_number,
            duration_secs=duration_secs,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=test_loss,
            epochs=epochs,
        )

        return self.insert_row("fold", data)
