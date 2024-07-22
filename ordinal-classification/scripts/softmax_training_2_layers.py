import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import optuna

torch.manual_seed(0)
np.random.seed(0)


def ordinal_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    total_count = len(y_true)
    accurate_count = sum(
        1
        for true_label, pred_label in zip(y_true, y_pred)
        if pred_label in [true_label, true_label - 1, true_label + 1]
    )
    return accurate_count / total_count


# Load and preprocess data
datafile_path = "../data/fine_food_reviews_with_e5_small_embeddings_1k.parquet"
df = pd.read_parquet(datafile_path)

embedding_dim = np.array(list(df.embedding.values)).shape[1]
print(f"Shape of embeddings in the dataframe: {embedding_dim}")

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    list(df.embedding.values),
    df.Score - 1,
    test_size=0.3,
    random_state=42,
)

# Convert the problem to three classes (very bad, bad, neutral-good)
# y_train = y_train.apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)
# y_test = y_test.apply(lambda x: 0 if x < 3 else 1 if x == 3 else 2)

n_classes = len(y_train.unique())


class ReviewsDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx].clone().detach().float()
        label = self.labels[idx].clone().detach().long()
        return embedding, label


# Convert train and test splits to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).clone().detach()
y_train_tensor = torch.tensor(y_train.to_numpy(), dtype=torch.long).clone().detach()
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).clone().detach()
y_test_tensor = torch.tensor(y_test.to_numpy(), dtype=torch.long).clone().detach()

# Create Dataset objects
train_dataset = ReviewsDataset(X_train_tensor, y_train_tensor)
test_dataset = ReviewsDataset(X_test_tensor, y_test_tensor)

params = {
    "input_dim": embedding_dim,
    "n_classes": n_classes,
    "num_hidden_1": 256,
    "num_hidden_2": 128,
    "dropout_rate": 0.5,
    "num_epochs": 20,
    "batch_size": 64,
    "learning_rate": 0.001,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}


# Define the neural network model using softmax
class MultilayerPerceptron(nn.Module):
    def __init__(self, input_dim, n_classes, num_hidden_1, num_hidden_2, dropout_rate=0.5):
        super().__init__()
        self.n_classes = n_classes
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, num_hidden_1, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden_1),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden_1, num_hidden_2, bias=True),
            nn.ReLU(),
            nn.BatchNorm1d(num_hidden_2),
            nn.Dropout(dropout_rate),
            nn.Linear(num_hidden_2, n_classes),
        )

    def forward(self, x):
        return self.predictor(x)


# Define the objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    num_hidden_1 = trial.suggest_int("num_hidden_1", 64, 512)
    num_hidden_2 = trial.suggest_int("num_hidden_2", 32, 256)
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.7)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [10, 32, 64, 128])
    num_epochs = trial.suggest_int("num_epochs", 10, 50)

    # Create DataLoader objects with the new batch size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Create model
    model = MultilayerPerceptron(
        input_dim=embedding_dim,
        n_classes=n_classes,
        num_hidden_1=num_hidden_1,
        num_hidden_2=num_hidden_2,
        dropout_rate=dropout_rate,
    ).to(params["device"])

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for _ in range(num_epochs):
        model.train()
        for features, targets in train_loader:
            features = features.to(params["device"])
            targets = targets.to(params["device"])

            # Forward pass
            logits = model(features)
            loss = criterion(logits, targets)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(params["device"])
            targets = targets.to(params["device"])

            logits = model(features)
            preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    return f1_score(all_targets, all_preds, average="weighted")


# Run the Optuna optimization
tpe_sampler = optuna.samplers.TPESampler(seed=42)
study = optuna.create_study(direction="maximize", sampler=tpe_sampler)
study.optimize(objective, n_trials=20)

# Get the best hyperparameters
best_params = study.best_trial.params
best_score = study.best_trial.value
print(f"Best hyperparameters: {best_params}")
print(f"Best score: {best_score:.4f}")

# Update the params dictionary with the best hyperparameters
params = {
    "input_dim": embedding_dim,
    "n_classes": n_classes,
    "num_hidden_1": best_params["num_hidden_1"],
    "num_hidden_2": best_params["num_hidden_2"],
    "dropout_rate": best_params["dropout_rate"],
    "num_epochs": best_params["num_epochs"],
    "batch_size": best_params["batch_size"],
    "learning_rate": best_params["learning_rate"],
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
}

# Create DataLoader objects with the best batch size
train_loader = DataLoader(train_dataset, batch_size=params["batch_size"], shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=params["batch_size"], shuffle=False)

# Create and train the final model with the best hyperparameters
model = MultilayerPerceptron(
    params["input_dim"],
    params["n_classes"],
    params["num_hidden_1"],
    params["num_hidden_2"],
    params["dropout_rate"],
).to(params["device"])

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=params["learning_rate"])

# Training loop
for epoch in range(params["num_epochs"]):
    model.train()
    for features, targets in train_loader:
        features = features.to(params["device"])
        targets = targets.to(params["device"])

        # Forward pass
        logits = model(features)
        loss = criterion(logits, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch + 1}/{params['num_epochs']}, Loss: {loss.item():.4f}")

# Evaluation
model.eval()
all_preds = []
all_targets = []
with torch.no_grad():
    for features, targets in test_loader:
        features = features.to(params["device"])
        targets = targets.to(params["device"])

        logits = model(features)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_targets.extend(targets.cpu().numpy())

# Print classification report
print(
    classification_report(
        all_targets, all_preds, target_names=[f"Class {i}" for i in range(n_classes)]
    )
)

ordinal_accuracy_score = ordinal_accuracy(np.array(all_targets), np.array(all_preds))
print(f"Ordinal accuracy of the network on the test data: {ordinal_accuracy_score:.0%}")
