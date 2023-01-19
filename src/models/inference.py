# import torch
# from torchvision import transforms
# from PIL import Image
# import os
# from pathlib import Path


# # load model
# dir_root = Path(__file__).parent.parent.parent
# model_path = Path(dir_root, './models/model_epochs10_lr1000.0_batch_size64.pth')
# image_path = Path(dir_root, './data/processed/images/apple_pie/134.jpg')
# print(model_path)
# print(image_path)
# #model = torch.load(model_path)
# # model.eval()
# model = torch.load(model_path, map_location=torch.device('cpu'))
# #model = model['model']


# # define image transform
# transform = transforms.Compose([transforms.Resize(256),
#                                 transforms.CenterCrop(224),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# # open image and transform
# img = Image.open(image_path)
# img_tensor = transform(img).unsqueeze(0)

# # run inference
# output = model(img_tensor)

# # get class with highest probability
# _, pred = output.max(1)

# # print class label
# print(pred.item())
import torch
from torchvision import models
from torchvision import transforms
from PIL import Image
from pathlib import Path

# load model
dir_root = Path(__file__).parent.parent.parent
model_path = Path(dir_root, "./models/model_epochs10_lr1000.0_batch_size64.pth")
image_path = Path(dir_root, "./data/processed/images/apple_pie/3670548.jpg")
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

print(class_dict)

model = models.resnet18(pretrained=True)
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 101)
model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))

# model = models.resnet18(pretrained=True)
# model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

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
img = Image.open(image_path)
img_tensor = transform(img).unsqueeze(0)

# run inference
with torch.no_grad():
    output = model(img_tensor)

# get class with highest probability
_, pred = output.max(1)

# print class label
pred_label = pred.item()
object_name = list(class_dict.keys())[list(class_dict.values()).index(pred_label)]
print(object_name)
