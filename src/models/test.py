from train_model import train
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torchvision import datasets, transforms
import torch

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
testset = datasets.MNIST('~/.pytorch/F_MNIST_data/', download=True, train=False, transform=transform)

train(trainset, testset)





