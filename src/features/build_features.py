from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from typing import Union
from PIL import Image

dir_root = Path(__file__).parent.parent.parent
dataset_raw_images = Path(dir_root, "./data/processed/images")
dataset_raw_labels = Path(dir_root, "./data/processed/meta/train.json")
dataset_raw_classes = Path(dir_root, "./data/processed/meta/classes.txt")
start_from_image: int = 0


def load_image(path):
    x = Image.open(path)
    # x = cv2.imread(str(path))
    # x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    return x


def prepare_data(num_images: int):
    """
    Function that loads the data. You can input the number of images
    you want to load.

            num_images: amount of images to be loaded (must be int)
            return: arrays X_train, X_val, y_train, y_val in this order, containing the train and validation split for feature and target vectors. The images of the X vector are stored as a path to the local file and not loaded yet.
    """

    image_load_count: Union[int, bool] = num_images
    df = pd.read_json(dataset_raw_labels)
    df_final = df.copy()

    for _class in df.columns:
        df_final[_class] = df_final.apply(
            lambda row: str(dataset_raw_images) + "/" + row[_class] + ".jpg", axis=1
        )

    df_final = df_final.melt(value_name="images")
    df_final.rename(columns={"variable": "label"}, inplace=True)

    with open(dataset_raw_classes, "r") as f:
        class_dict = dict()
        i = 0
        for line in f:
            x = line.strip("\n").split(" ")
            key = x[0]
            value = i
            i = i + 1
            if key not in class_dict.keys():
                class_dict[key] = value
            else:
                class_dict[key].append(value)

    # print(class_dict)

    if image_load_count != False:
        idxs = df_final.index.to_list()
        delete_before = idxs[:(start_from_image)]
        delete_after = idxs[(start_from_image + image_load_count) :]

        df_final.drop(delete_after, axis=0, inplace=True)
        df_final.drop(delete_before, axis=0, inplace=True)
        df_final.reset_index(inplace=True)

    df_final["label"] = df_final["label"].apply(lambda x: class_dict[x])
    # df_final['images'] = df_final['images'].apply(lambda row:  load_image(row))
    df_final.drop(["index"], axis=1)

    X_train, X_val, y_train, y_val = train_test_split(
        df_final["images"], df_final["label"], test_size=0.2, random_state=42
    )

    # train_dataset = FoodDataset(X_train, y_train)
    # train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

    # val_dataset = FoodDataset(X_val, y_val)
    # val_dataloader = DataLoader(val_dataset, batch_size=batchsize, shuffle=True)

    return X_train, X_val, y_train, y_val


preprocess = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


class FoodDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images.index)

    def __getitem__(self, idx):
        return preprocess(load_image(self.images.iloc[idx])), self.labels.iloc[idx]
