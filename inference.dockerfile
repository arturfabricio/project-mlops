FROM python:3.9.11-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN pip install torch torchvision numpy Pillow

# Test
COPY src/ src/
COPY models/model_epochs10_lr1000.0_batch_size64.pth models/model_epochs10_lr1000.0_batch_size64.pth
COPY data/processed/meta/classes.txt data/processed/meta/classes.txt
COPY data/processed/images/churros/1601.jpg data/processed/images/apple_pie/3670548.jpg

WORKDIR /

ENTRYPOINT ["python", "-u", "src/models/inference.py"]