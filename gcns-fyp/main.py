import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
from ax import optimize

from models import GCN

dataset = TUDataset(root='data/TUDataset', name='MUTAG')

torch.manual_seed(12345)
dataset = dataset.shuffle()

train_dataset = dataset[:150]
test_dataset = dataset[150:]

print(f'Number of training graphs: {len(train_dataset)}')
print(f'Number of test graphs: {len(test_dataset)}')


train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
criterion = torch.nn.CrossEntropyLoss()

def train(optimiser, model):
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimiser.step()  # Update parameters based on gradients.
         optimiser.zero_grad()  # Clear gradients.


def test(loader, model):
     model.eval()

     correct = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     return correct / len(loader.dataset)  # Derive ratio of correct predictions.

def training_run(params):
    model = GCN(
        hidden_channels=2**params["channels_power_of_two"], 
        num_node_features=dataset.num_node_features, 
        num_classes=dataset.num_classes
    )
    optimiser = torch.optim.Adam(model.parameters(), lr=params["lr"])

    for epoch in range(1, params["epochs"]):
        train(optimiser, model)
        train_acc = test(train_loader, model)
        test_acc = test(test_loader, model)
        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    print("training run complete")
    print("params", params)
    print("accuracy", test_acc)
    return test_acc

# Define a search space
results = optimize(
    parameters=[
        dict(name="lr", bounds=[0.000001, 1.0], type="range", log_scale=True, value_type="float"),
        dict(name="channels_power_of_two", bounds=[2, 9], value_type="int", type="range"),
        dict(name="epochs", value=101, value_type="int", type="fixed")
    ],
    evaluation_function=training_run
)
print(results)

