import os
import sys
import torch

data_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.append(os.path.abspath(data_path))

# Import the data module
from build_features import prepare_data

def test_data():

    num_images = 2000
    batchsize = 128
    #assert the number of batch sizes compared to the number of images
    train_data, val_data = prepare_data(num_images=num_images, batchsize=batchsize)
    assert (len(train_data) == int(0.8*num_images/batchsize))
    assert (len(val_data) == int(0.2*num_images/batchsize))

      
    #assert that each datapoint has shape [3,224,224]
    bool = True 
    for images, labels in train_data:
        for image in images:
            bool and (image.shape == torch.Size([3, 224, 224]))
    assert bool   

    #assert that each label is represented == (3,256,256)


    
    
test_data()    
 