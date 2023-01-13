import timm 
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
import os
import sys

data_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.append(os.path.abspath(data_path))
from build_features import prepare_data

#Defining the validation loss calculation that will be used in the training function
def compute_validation_metrics(model, dataloader):
    ''' Gives the loss and the accuracy of a model one a certain test set'''
    model.eval()
    total_loss = 0
    correct = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            output_raw = model(inputs)
            output = torch.nn.LogSoftmax(dim=1)(output_raw)
            loss = torch.nn.functional.nll_loss(output,labels)
            total_loss += loss.item()
            _, preds = torch.max(input=output, dim=1)            
            correct += (preds == labels).sum().item()
    return total_loss / len(dataloader), 100. * correct / len(dataloader.dataset)

# Training function
def main(chosen_model='resnet18', batch_size=64, epochs=5, lr=0.001, num_images=100):
    ''' Trains a neural network from the TIMM framework'''
    
    print("Start training with: " + chosen_model)
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on: ", DEVICE)

    train_loader, val_loader = prepare_data(num_images,batch_size)
    model = timm.create_model(chosen_model, pretrained=True, num_classes = 101)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)               

    for epoch in range(epochs):
        print("Epoch {i}/{j}...".format(i=epoch+1, j=epochs))
        overall_loss = 0
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            output_raw = model(inputs)
            output = torch.nn.LogSoftmax(dim=1)(output_raw)
            loss = torch.nn.functional.nll_loss(output,targets)
            loss.backward()
            optimizer.step()
            overall_loss += loss.item()

        train_loss, train_acc = compute_validation_metrics(model,train_loader)
        print('train loss: {tl} '.format(tl=train_loss), 'train accuracy: {ta}'.format(ta=train_acc))
        val_loss, val_acc = compute_validation_metrics(model,val_loader)
        print('validation loss: {vl} '.format(vl=val_loss), 'validation accuracy: {va}'.format(va=val_acc))
        print('Average loss for epoch {i}: {loss}'.format(i=epoch+1, loss=overall_loss/len(train_loader)))
    return model  

