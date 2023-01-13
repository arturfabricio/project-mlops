# FROM python:3.9.11-slim

# # install python 
# RUN apt update && \
#     apt install --no-install-recommends -y build-essential gcc && \
#     apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY src/ src/
# WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
# RUN dvc init
# RUN dvc pull

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

# Use a base image with Python 3.9.11
FROM python:3.9.11-slim

# install build-essential and gcc
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# install google cloud sdk
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get install apt-transport-https ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# authenticate with google cloud
COPY service_account.json service_account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/service_account.json
RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# copy requirements.txt and project files
COPY requirements.txt requirements.txt
COPY src/ src/

# install python dependencies
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# copy data from GCS bucket
RUN gsutil -m cp -r gs://artifacts.mlopsproject-374511.appspot.com/* /data/

# run the training script
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]