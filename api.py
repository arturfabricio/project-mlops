
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse
import cv2
from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
import torch
from torchvision import models

app = FastAPI()

dir_root = Path(__file__).parent
dataset_raw_classes = Path(dir_root, './data/processed/meta/classes.txt')
model_path = Path(dir_root, './models/model_epochs10_lr1000.0_batch_size64.pth')


def load_class_dict(path): 
    
    with open(path, "r") as f:
        class_dict = dict()
        i = 0
        for line in f:
            x = line.strip("\n").split(" ")
            key = x[0]
            value = i
            i = i+1
            if key not in class_dict.keys():
                class_dict[key] = value
            else:
                class_dict[key].append(value)
    return class_dict

@app.post("/predict/")
async def cv_model(image_file: UploadFile = File(...)):

    transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.226])])

    img = Image.open(image_file.file)
    img_tensor = transform(img).unsqueeze(0)

    img_np = np.array(img_tensor[0].permute(1, 2, 0))
    cv2.imwrite('image_resize.jpg', img_np)
    image_file.file.close()

    class_dict = load_class_dict(dataset_raw_classes)

    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 101)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.eval()
    
    with torch.no_grad():
        output = model(img_tensor)

    # get class with highest probability
    _, pred = output.max(1)

    # print class label
    pred_label = pred.item()
    object_name = list(class_dict.keys())[list(class_dict.values()).index(pred_label)]

    
    return FileResponse('image_resize.jpg'), object_name