import timm 
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader



# Model Hyperparameters
dataset_path = 'datapath'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 100
epochs = 1
lr = 0.01

# Training function
def train ():
    print("Start training ResNet...")

    model = timm.create_model('resnet18', pretrained=False)
    model.to(DEVICE)

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        print("Epoch 1/{i}...".format(i=epochs))
        overall_loss = 0
        for images, labels in trainloader:
            
            optimizer.zero_grad()
            output = model(images)
            loss = loss_fn(output,labels)
            loss.backward()
            optimizer.step()

            overall_loss += loss.item()

        print('Overall loss for this epoch : {i}'.format(i=overall_loss))
      

train()




