# Project description:

*Developped by:* \
César Delafargue (s212834) \
Artur C. Fabrício (s213242)\
Lucas Lyck (s183685)

- **Overall goal of the project**: The goal of this project is to implement a reproducible machine learning model, which can recognize different types of food in pictures. 

- **What framework are you going to use (PyTorch Image Models, Transformer, Pytorch-Geometrics):** We intend on using [PyTorch Image Models](https://github.com/rwightman/pytorch-image-models) as our framework, since our problem is in the field of Computer Vision.

- **How to you intend to include the framework into your project:** The goal of the framework is to accelerate our project by providing ready-to-use pre-trained models that we can import directly in our project and train on the data we chose. We can thus avoid losing time by doing this ourselves, and focus on improving other aspects of the code.

- **What data are you going to run on (initially, may change):** The data used for this this project will initially be based on the Food-101 Data set. The dataset is consisting of 101 food categories with a total of 101000 images. For each class there are 1000 images including 250 manually reviewed test images and 750 training images. All images were rescaled to have a maximum side length of 512 pixels. The dataset download link and documentation can be found [here](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/).

- **What deep learning models do you expect to use:** Since we're using a framework that has quite a lot of models available, we hope to attempt training with various popular models, and in the end perform a comparison between them. All these models have pretrained weights that we plan on using, due to the reduced time available for the project. We also hope to improve their performance, and throughly document the results, with tools such as [wandb](https://wandb.ai/site). Some of the popular models we expect to train (for now) include:

    - [DenseNet](https://arxiv.org/abs/1608.06993)
    - [ResNet](https://arxiv.org/abs/1512.03385)
    - [VGG](https://arxiv.org/pdf/1409.1556.pdf)

## Tools and Features Implemented in the Project:

- **Data Version Control & Google Bucket Storage:** all our data is stored in a [Google Bucket](https://console.cloud.google.com/storage/browser/dtu-mlops-bucket-project), and fully integrated with DVC, meaning that with a simple `dvc pull`, all the data is easily downloadable. 

- **Weights and Biases Sweep and Logging:** we provide a version of our training function, `main()`, where in which we take advantage of the `wandb` platform to log the results of our training, as well as performing a hyperparameter sweeping that allows us to check for the best hyperparameters for our model. To perform a training session with `wandb`, use [src/models/sweep_model.py](https://github.com/arturfabricio/project-mlops/blob/main/src/models/sweep_model.py).


Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
