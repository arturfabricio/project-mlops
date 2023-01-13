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

FROM python:3.9.11-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc dvc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY src/ src/
WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir

# install google cloud sdk
RUN echo "deb [signed-by=/usr/share/keyrings/cloud.google.gpg] https://packages.cloud.google.com/apt cloud-sdk main" | tee -a /etc/apt/sources.list.d/google-cloud-sdk.list && \
    apt-get install apt-transport-https ca-certificates && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key --keyring /usr/share/keyrings/cloud.google.gpg  add - && \
    apt-get update -y && apt-get install google-cloud-sdk -y

# authenticate with google cloud
COPY service_account.json service_account.json
ENV GOOGLE_APPLICATION_CREDENTIALS=/service_account.json
RUN gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# initialize DVC repository
RUN dvc init

# add GCS as a remote storage
RUN dvc remote add -d myremote gs://artifacts.mlopsproject-374511.appspot.com

# add data to dvc
RUN dvc add data/
RUN dvc push

# pull data from remote storage
RUN dvc pull data/

# copy data to the container
COPY data/ /data/

# run train script
ENTRYPOINT ["python", "-u", "src/models/train_model.py"]