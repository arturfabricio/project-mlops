import timm 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb
import os
import sys

data_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.append(os.path.abspath(data_path))

# Import the data module
from build_features import prepare_data

#Defining the validation loss calculation that will be used in the training function
def compute_validation_metrics(model, dataloader, loss_fn):
    ''' Gives the loss and the accuracy of a model one a certain test set'''
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            probabilities = torch.nn.Softmax(dim=1)(outputs)
            loss = loss_fn(probabilities, labels)
            total_loss += loss.item()
            _, preds = torch.max(input=probabilities, dim=1)
            total_acc += (preds == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader)


# Training function
def train (chosen_model='resnet18', batch_size=64, epochs=2, lr=0.001, num_images=400):
    ''' Trains a neural network from the TIMM framework'''
    
    print("Start training with: " + chosen_model)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Running on: ", DEVICE)

    train_loader, val_loader = prepare_data(num_images,batch_size)
    print("Training batches loaded: ", len(train_loader))
    print("Validation batches loaded: ", len(val_loader))

    model = timm.create_model(chosen_model, pretrained=True, num_classes = 101)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch {i}/{j}...".format(i=epoch+1, j=epochs))
        overall_loss = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
           
            optimizer.zero_grad()

            output_raw = model(inputs)
            output = torch.nn.Softmax(dim=1)(output_raw)
            loss = loss_fn(output,targets)

            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        train_loss, train_acc = compute_validation_metrics(model,train_loader,loss_fn)
        print('train loss: {tl} '.format(tl=train_loss), 'train accuracy: {ta}'.format(ta=train_acc))
        val_loss, val_acc = compute_validation_metrics(model,val_loader,loss_fn)
        print('validation loss: {vl} '.format(vl=val_loss), 'validation accuracy: {va}'.format(va=val_acc))

        # wandb.log({
        #     'epoch': epoch, 
        #     'train_acc': train_acc,
        #     'train_loss': train_loss, 
        #     'val_acc': val_acc, 
        #     'val_loss': val_loss
        # })
        
        print('Average loss for epoch {i}: {loss}'.format(i=epoch, loss=overall_loss/len(train_loader)))
    
    return model  

train()




