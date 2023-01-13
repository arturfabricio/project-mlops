FROM python:3.9.11-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY data/ data/
COPY src/ src/
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN dvc pull data/
COPY data/ /data/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

