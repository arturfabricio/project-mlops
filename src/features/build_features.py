from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2 as cv2
from PIL import Image
from torchvision import transforms

dir_root = Path(__file__).parent.parent.parent
dataset_raw_images = Path(dir_root, './data/processed/images')
dataset_raw_labels = Path(dir_root, './data/processed/meta/train.json')

def load_image(path):
    try:
        x = cv2.imread(str(path))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)  
        return x
    except:
        print('Image doesnt exist')
        return 0

def prepare_data():
    used_classes = ['apple_pie','baby_back_ribs','beef_tartare', 
                'caesar_salad','carrot_cake','chicken_wings', 'donuts',
                'french_fries','grilled_salmon','lasagna','omelette',
                'pizza','prime_rib']

    df = pd.read_json(dataset_raw_labels)
    df_final = df[used_classes]

    for _class in used_classes: 
        df_final[_class] = df_final.apply(lambda row: str(dataset_raw_images) + "/" + row[_class] + '.jpg', axis=1)

    # df_final = df_final.stack()
    df_final = df_final.melt(value_name='images')
    df_final.rename(columns = {'variable':'label'}, inplace = True)
    
    class_dict = {'apple_pie':0,
                  'baby_back_ribs':1,
                  'beef_tartare':2,  
                  'caesar_salad':3,
                  'carrot_cake':4,
                  'chicken_wings':5,
                  'donuts':6,          
                  'french_fries':7,
                  'grilled_salmon':8,
                  'lasagna':9,
                  'omelette':10,        
                  'pizza':11,
                  'prime_rib':12}

    print(df_final.head())
    df_final['label'] = df_final['label'].apply(lambda x:  class_dict[x])
    df_final['images'] = df_final['images'].apply(lambda row:  load_image(row))
    print(df_final.head())
    df_final.drop(df_final.loc[df_final['images']==0], inplace=True)
    print(df_final.head())


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
    # train_dataloader = FoodDataset(train_dataset, batch_size=64, shuffle=True)

    # test_dataset = FoodDataset(test_images, test_labels)
    # test_dataloader = FoodDataset(test_dataset, batch_size=64, shuffle=True)

    # return train_dataloader, test_dataloader

prepare_data()
