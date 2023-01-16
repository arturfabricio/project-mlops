
# # from fastapi import FastAPI, File, UploadFile
# # from PIL import Image
# # import io

# # app = FastAPI()

# # @app.post("/resize/{size}")
# # async def resize(file: UploadFile, size: int):
# #     image = Image.open(io.BytesIO(await file.read()))
# #     image = image.resize((size, size))
# #     return image

from fastapi import FastAPI, File, UploadFile
from fastapi import UploadFile, File
from fastapi.responses import FileResponse
import cv2
from typing import Optional
from http import HTTPStatus
import torch
import numpy as np

app = FastAPI()

@app.post("/resize/")
async def cv_model(model: UploadFile, image_file: UploadFile = File(...)):

    model = torch.load(model.file)


    # with open('image.jpg', 'wb') as image:
    #     content = await image_file.read()
    #     image.write(content)
    #     image.close()

    # img = cv2.imread("image.jpg")
    # res = cv2.resize(img, (h, w))

    # cv2.imwrite('image_resize.jpg', res)

    image = cv2.imdecode(np.frombuffer(await image_file.read(), np.uint8), cv2.IMREAD_COLOR)
    image = cv2.resize(image, (224, 224))

    response = {
        "input": image_file,
        "output": FileResponse('image_resize.jpg'),
        "message": HTTPStatus.OK.phrase,
        "status-code": HTTPStatus.OK,
    }
    
    with torch.no_grad():
        output = model(img)
    _, predicted = torch.max(output, 1)
    return response, {"predicted_class": predicted.item()}

# from fastapi import FastAPI, File, UploadFile
# from PIL import Image

# app = FastAPI()

# @app.post("/resize_and_operation/")
# async def resize_and_operation(file: UploadFile, operation: str):
#     # Open the image
#     with Image.open(file.file) as img:
#         # Resize the image to 224x224
#         img = img.resize((224, 224))
        
#         # Perform the desired operation
#         if operation == "grayscale":
#             img = img.convert("L")
#         elif operation == "rotate":
#             img = img.rotate(45)
#         # add your other operation here
        
#         # Save the modified image
#         img_bytes = img.tobytes()
#     return {"image_bytes": img_bytes}
