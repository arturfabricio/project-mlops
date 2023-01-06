import timm 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import wandb

#Defining the validation loss calculation that will be used in the training function
def compute_validation_metrics(model, dataloader, loss_fn):
    ''' Gives the loss and the accuracy of a model one a certain test set'''
    model.eval()
    total_loss = 0
    total_acc = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
    return total_loss / len(dataloader), total_acc / len(dataloader)

# Training function
def train (train_dataset, test_dataset, chosen_model = 'resnet18', batch_size = 128, epochs = 5, lr = 0.01):
    ''' Trains a neural network from the TIMM framework'''
    
    print("Start training " + chosen_model)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=False)

    model = timm.create_model(chosen_model, pretrained=True, num_classes=12)
    model.to(DEVICE)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        print("Epoch {i}/{j}...".format(i=epoch, j=epochs))
        overall_loss = 0
        for images, labels in train_loader:
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        train_loss, train_acc = compute_validation_metrics(model,train_loader)
        val_loss, val_acc = compute_validation_metrics(model,test_loader)

        wandb.log({
            'epoch': epoch, 
            'train_acc': train_acc,
            'train_loss': train_loss, 
            'val_acc': val_acc, 
            'val_loss': val_loss
        })

        print('Average loss for epoch : {i}'.format(i=overall_loss/len(train_loader)))
    
    return model  





