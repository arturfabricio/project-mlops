import os
import sys

import timm
import torch
import wandb
from torch.optim import Adam
from torch.utils.data import DataLoader

wandb.login()

data_path = os.path.join(os.path.dirname(__file__), "../features")
sys.path.append(os.path.abspath(data_path))
from build_features import FoodDataset, prepare_data


# Defining the validation loss calculation that will be used in the training function
def compute_validation_metrics(model, dataloader):
    """ Gives the loss and the accuracy of a model one a certain test set"""
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            output_raw = model(inputs)
            output = torch.nn.LogSoftmax(dim=1)(output_raw)
            loss = torch.nn.functional.nll_loss(output, labels)
            total_loss += loss.item()
            _, preds = torch.max(input=output, dim=1)
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), 100.0 * correct / len(dataloader.dataset)


# Training function
def main(chosen_model="resnet18", batch_size=64, epochs=5, lr=0.001, num_images=500):
    """ Trains a neural network from the TIMM framework (with sweep using wandb)"""

    print("Start training with: " + chosen_model)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", DEVICE)

    run = wandb.init(project="my-first-sweep")
    lr = wandb.config.lr
    batch_size = wandb.config.batch_size
    epochs = wandb.config.epochs

    X_train, X_val, y_train, y_val = prepare_data(num_images)
    train_dataset = FoodDataset(X_train, y_train)
    val_dataset = FoodDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )

    model = timm.create_model(chosen_model, pretrained=True, num_classes=101)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        print("Epoch {i}/{j}...".format(i=epoch + 1, j=epochs))
        overall_loss = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output_raw = model(inputs)
            output = torch.nn.LogSoftmax(dim=1)(output_raw)
            loss = torch.nn.functional.nll_loss(output, targets)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

        train_loss, train_acc = compute_validation_metrics(model, train_loader)
        print(
            "train loss: {tl} ".format(tl=train_loss),
            "train accuracy: {ta}".format(ta=train_acc),
        )
        val_loss, val_acc = compute_validation_metrics(model, val_loader)
        print(
            "validation loss: {vl} ".format(vl=val_loss),
            "validation accuracy: {va}".format(va=val_acc),
        )

        wandb.log(
            {
                "epoch": epoch,
                "train_acc": train_acc,
                "train_loss": train_loss,
                "val_acc": val_acc,
                "val_loss": val_loss,
            }
        )

        print(
            "Average loss for epoch {i}: {loss}".format(
                i=epoch + 1, loss=overall_loss / len(train_loader)
            )
        )
    return model


sweep_configuration = {
    "method": "random",
    "name": "sweep",
    "metric": {"goal": "maximize", "name": "val_acc"},
    "parameters": {
        "batch_size": {"values": [16, 32, 64]},
        "epochs": {"values": [5, 20, 50]},
        "lr": {"max": 0.1, "min": 0.0001},
    },
}

sweep_id = wandb.sweep(sweep=sweep_configuration, project="my-first-sweep")
wandb.agent(sweep_id, function=main, count=4)
