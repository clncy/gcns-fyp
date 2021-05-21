CREATE TABLE IF NOT EXISTS experiment (
    id INTEGER PRIMARY KEY,
    description TEXT,
    datetime TEXT, -- ISO8601
    model TEXT,
    dataset TEXT,
    trials INTEGER,
    training_split REAL,
    optimisation_strategy TEXT,
    max_epochs INTEGER,
    optimiser TEXT,
    metric TEXT
);

CREATE TABLE IF NOT EXISTS trial (
    id INTEGER PRIMARY KEY,
    experiment_id INTEGER REFERENCES experiment(id),
    trial_number INTEGER,
    learning_rate REAL,
    batch_size INTEGER,
    hidden_channels INTEGER,
    dropout_probability REAL,
    representation_size INTEGER,
    hidden_layers INTEGER,
    embedding_dimensions INTEGER
);

CREATE TABLE IF NOT EXISTS fold (
    id INTEGER PRIMARY KEY,
    trial_id INTEGER REFERENCES trial(id),
    fold_number INTEGER,
    duration_secs REAL,
    train_loss REAL,
    val_loss REAL,
    test_loss REAL,
    epochs INT
);
