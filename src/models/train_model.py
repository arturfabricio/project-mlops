import os
from pathlib import Path

import timm
import torch
from torch.optim import Adam
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler
from torch.utils.data import DataLoader

from src.features.build_features import FoodDataset, prepare_data

dir_root = Path(__file__).parent.parent.parent
print(dir_root)


with_profile = False  ## Will have to add this to a click guard

# @click.group()
# def cli():
#     pass

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


# @click.command()
# @click.option("--lr", default=1e-3, help='learning rate to use for training')
# @click.option("--batch_size", default=64, help='learning rate to use for training')
# @click.option("--epochs", default=10, help='number of epcohs to use for training' )
# @click.option("--mdl", default='resnet18', help='model to be used')
# @click.option("--num_images",default=100, help="Number of images to use")
# @click.option("--save_model",default=False, help="Define if model should be saved (False=not save; True=save)")
def main(
    mdl="resnet18",
    batch_size=64,
    epochs=10,
    lr=1e-3,
    num_images=100,
    save_model=True
):
    """ Trains a neural network from the TIMM framework"""

    print("Start training with: " + mdl)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", DEVICE)

    X_train, X_val, y_train, y_val = prepare_data(num_images)
    train_dataset = FoodDataset(X_train, y_train)
    val_dataset = FoodDataset(X_val, y_val)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=3
    )

    model = timm.create_model(mdl, pretrained=True, num_classes=101)
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
        print(
            "Average loss for epoch {i}: {loss}".format(
                i=epoch + 1, loss=overall_loss / len(train_loader)
            )
        )
        # prof.step()
    if save_model:
        pth = f"models/model_epochs{epochs}_lr{lr}_batch_size{batch_size}.pth"
        torch.save(
            model.state_dict(),
            os.path.join(
                dir_root,
                pth,
            ),
        )

    return model


# cli.add_command(main)

# if __name__ == "__main__":
#     cli()

if __name__ == "__main__":

    if with_profile:
        with profile(
            activities=[ProfilerActivity.CPU],
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler("./log/model"),
        ) as prof:
            main(epochs=2)
        print(
            prof.key_averages(group_by_input_shape=True).table(
                sort_by="cpu_time_total", row_limit=30
            )
        )
    else:
        main(
            mdl="resnet18",
            batch_size=128,
            epochs=2,
            lr=1e-2,
            num_images=2000,
            save_model=False,
        )
