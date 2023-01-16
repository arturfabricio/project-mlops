FROM python:3.9.11-slim

# install python 
RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

# COPY requirements.txt requirements.txt
# COPY src/ src/
# COPY .dvc/ .dvc/
# COPY .dvcignore .dvcignore
# COPY .github/ .github/
# COPY .gitignore .gitignore
RUN apt-get update && apt-get install -y git
RUN git clone https://github.com/arturfabricio/project-mlops
RUN ls
WORKDIR /project-mlops
RUN git pull --rebase https://github.com/arturfabricio/project-mlops

RUN pip install -r requirements.txt --no-cache-dir

# RUN pip install dvc
# RUN pip install "dvc[all]"
# RUN ls
# RUN dvc pull

# COPY data/ data/

ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

