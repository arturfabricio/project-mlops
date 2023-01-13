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
COPY project-mlops/ project-mlops/
WORKDIR /
# RUN pip install -r requirements.txt --no-cache-dir
RUN pip install dvc
RUN pip install "dvc[all]"
RUN ls
# RUN cd project-mlops
# RUN dvc init --no-scm
RUN dvc pull
# COPY data/ data/

# ENTRYPOINT ["python", "-u", "src/models/train_model.py"]

