import requests
from pathlib import Path


url = "http://localhost:8000/predict/"

dir_root = Path(__file__).parent

image_path = Path(dir_root, "./data/processed/images/apple_pie/3670548.jpg")

with open(image_path, "rb") as f:
    data = {"image_file": f}
    response = requests.post(url, files=data)
    print(response.text)
