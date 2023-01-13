import os
import sys
import torch
import pytest


data_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.append(os.path.abspath(data_path))


# Import the data module

from build_features import prepare_data

@pytest.mark.skipif(not os.path.exists('data/'), reason="Data files not found")
def test_data():

    num_images = 2000
    batchsize = 128
    #assert the number of batch sizes compared to the number of images
    train_data, val_data = prepare_data(num_images=num_images, batchsize=batchsize)
    assert (len(train_data) == int(0.8*num_images/batchsize + 1)), "The train batches don't match the amount of data points"
    assert (len(val_data) == int(0.2*num_images/batchsize + 1)), "The test batches don't match the amount of data points"

      
    #assert that each datapoint has shape [3,224,224]

    bool = True 
    for images, labels in train_data:
        for image in images:
            bool and (image.shape == torch.Size([3, 224, 224]))
    assert bool, "The images don't have the right format" 

    #assert that each label is represented == (3,256,256)

