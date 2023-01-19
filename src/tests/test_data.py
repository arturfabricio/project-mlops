import os
import sys
import torch
import pytest


data_path = os.path.join(os.path.dirname(__file__), "../features")
sys.path.append(os.path.abspath(data_path))


# Import the data module

from build_features import prepare_data, FoodDataset


@pytest.mark.skipif(
    not os.path.exists("data/"),
    reason="Data files not found, this should be tested locally only",
)
def test_data():

    num_images = 2000

    X_train, X_val, y_train, y_val = prepare_data(num_images)
    train_dataset = FoodDataset(X_train, y_train)
    val_dataset = FoodDataset(X_val, y_val)

    # assert the number of batch sizes compared to the number of images
    assert len(train_dataset) == int(
        num_images * 0.8
    ), "The train batches don't match the amount of data points"
    assert len(val_dataset) == int(
        num_images * 0.2
    ), "The test batches don't match the amount of data points"

    # assert that each datapoint has shape [3,224,224]

    bool = True
    for image, _ in train_dataset:
        bool and (image.shape == torch.Size([3, 224, 224]))
    assert bool, "The images don't have the right format"

    # assert that we have the right number of labels compared to the number of imported images

    labels_set = set()
    for _, label in train_dataset:
        labels_set.add(label.item())
    assert len(labels_set) == num_images // 1000 + 1
