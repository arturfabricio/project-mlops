import torch
import torch.nn as nn
import timm
from train_model import train

def test_training_weights():
    
    model = timm.create_model('resnet18', pretrained=True, num_classes = 12)
    print(model.layer1[0].conv1.weight)
    
    train_dataset = [(torch.randn(3,100,100),torch.randint(12,size=(1,)).item()) for i in range (10*128)] 

    trained_model = train(train_dataset, train_dataset, epochs=1)

    print(trained_model.layer1[0].conv1.weight)
    
   
test_training_weights()   