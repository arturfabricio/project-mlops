import os
import sys

data_path = os.path.join(os.path.dirname(__file__), '../features')
sys.path.append(os.path.abspath(data_path))

# Import the data module
from build_features import prepare_data

def test_data():

    num_images = 400
    #assert the size of training and validation sets
    train_data, val_data = prepare_data(num_images=num_images, batchsize=128)
    print(train_data.shape, val_data.shape)  
    assert train_data.shape[0] == 0.8*num_images
    assert val_data.shape[0] == 0.2*num_images
      
    #assert that each datapoint has shape [3,256,256]
    shapes=[] 
    for image in train_data:
        shapes.append(image.shape == (3,256,256))
    print (shapes)
    result = True
    for bool in shapes:
        result and bool
    print(result)    

    #assert that each label is represented


    
    
test_data()    
 