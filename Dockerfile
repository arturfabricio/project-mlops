FROM python:3.9.11-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*
# Test
COPY requirements.txt requirements.txt
COPY src/ src/
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
# RUN pip install -r requirements.txt --no-cache-dir && \
#     pip install -e .

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

