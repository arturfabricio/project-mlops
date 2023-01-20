from pathlib import Path

import torch
from PIL import Image
from torchvision import models, transforms

# load model
dir_root = Path(__file__).parent.parent.parent
model_pth = Path(dir_root, "./models/model_epochs10_lr1000.0_batch_size64.pth")
image_pth = Path(dir_root, "./data/processed/images/churros/1601.jpg")
dataset_raw_classes = Path(dir_root, "./data/processed/meta/classes.txt")


def load_class_dict(path):

    with open(path, "r") as f:
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
    return class_dict


class_dict = load_class_dict(dataset_raw_classes)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 101)
model.load_state_dict(torch.load(model_pth, map_location=torch.device("cpu")))

model.eval()

# define image transform
transform = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.226]),
    ]
)

# open image and transform
img = Image.open(image_pth)
img_tensor = transform(img).unsqueeze(0)

# run inference
with torch.no_grad():
    output = model(img_tensor)

# get class with highest probability
_, pred = output.max(1)

# print class label
pred_label = pred.item()
index_key = list(class_dict.values()).index(pred_label)
object_name = list(class_dict.keys())[index_key]
print(object_name)

# from pathlib import Path
# import argparse

# # parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument("--image_path", required=True)
# args = parser.parse_args()
# image_pth = args.image_path

# import torch
# from PIL import Image
# from torchvision import models, transforms

# # load model
# dir_root = Path(__file__).parent.parent.parent
# model_pth = Path(dir_root, "./models/model_epochs10_lr1000.0_batch_size64.pth")
# dataset_raw_classes = Path(dir_root, "./data/processed/meta/classes.txt")


# def load_class_dict(path):

#     with open(path, "r") as f:
#         class_dict = dict()
#         i = 0
#         for line in f:
#             x = line.strip("\n").split(" ")
#             key = x[0]
#             value = i
#             i = i + 1
#             if key not in class_dict.keys():
#                 class_dict[key] = value
#             else:
#                 class_dict[key].append(value)
#     return class_dict


# class_dict = load_class_dict(dataset_raw_classes)

# model = models.resnet18(pretrained=True)
# num_ftrs = model.fc.in_features
# model.fc = torch.nn.Linear(num_ftrs, 101)
# model.load_state_dict(torch.load(model_pth, map_location=torch.device("cpu")))

# model.eval()

# # define image transform
# transform = transforms.Compose(
#     [
#         transforms.Resize(256),
#         transforms.CenterCrop(224),
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.226]),
#     ]
# )

# # open image and transform
# img = Image.open(image_pth)
# img_tensor = transform(img).unsqueeze(0)

# # run inference
# with torch.no_grad():
#     output = model(img_tensor)

# # get class with highest probability
# _, pred = output.max(1)

# # print class label
# pred_label = pred.item()
# index_key = list(class_dict.values()).index(pred_label)
# object_name = list(class_dict.keys())[index_key]
# print(object_name)