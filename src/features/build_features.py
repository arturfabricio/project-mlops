from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt

def load_image(dir):
    return plt.imread(dir)

def prepare_data():
    dir_root = Path(__file__).parent.parent.parent
    dataset_raw_images = Path(dir_root, './data/processed/images')
    dataset_raw_labels = Path(dir_root, './data/processed/meta/train.json')
    
    used_classes = ['apple_pie','baby_back_ribs','beef_tartare', 
                'caesar_salad','carrot_cake','chicken_wings', 'donuts',
                'french_fries','grilled_salmon','lasagna','omelette',
                'pizza','prime_rib']

    df = pd.read_json(dataset_raw_labels)
    df1 = df[used_classes]
    df_final = pd.DataFrame(['images','labels'])

    print(df1.head())

    df_final['images'] = df1.apply(lambda column: load) 

    # with np.load(test_path) as data:
    #     test_images = data['images']
    #     test_labels = data['labels']
    
    # train_images = np.concatenate((train_images_1, train_images_2, train_images_3, train_images_4, train_images_5))
    # train_labels = np.concatenate((train_labels_1, train_labels_2, train_labels_3, train_labels_4, train_labels_5))

    class FoodDataset(Dataset):
        def __init__(self, images, labels):
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # train_dataset = FoodDataset(train_images, train_labels)
    # train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # test_dataset = FoodDataset(test_images, test_labels)
    # test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)

    # return train_dataloader, test_dataloader

prepare_data()
