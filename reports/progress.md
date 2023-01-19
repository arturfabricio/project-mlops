## Project checklist

Please note that all the lists are *exhaustive* meaning that I do not expect you to have completed very
point on the checklist for the exam.

### Week 1

* [X] Create a git repository
* [X] Make sure that all team members have write access to the github repository
* [X] Create a dedicated environment for you project to keep track of your packages (using conda)
* [X] Create the initial file structure using cookiecutter
* [X] Fill out the `make_dataset.py` file such that it downloads whatever data you need and
* [X] Add a model file and a training script and get that running
* [X] Remember to fill out the `requirements.txt` file with whatever dependencies that you are using
* [ ] Remember to comply with good coding practices (`pep8`) while doing the project
* [X] Do a bit of code typing and remember to document essential parts of your code
* [X] Setup version control for your data or part of your data
* [X] Construct one or multiple docker files for your code
* [X] Build the docker files locally and make sure they work as intended
* [ ] Write one or multiple configurations files for your experiments
* [ ] Used Hydra to load the configurations and manage your hyperparameters
* [ ] When you have something that works somewhat, remember at some point to to some profiling and see if
      you can optimize your code
* [ ] Use wandb to log training progress and other important metrics/artifacts in your code
* [ ] Use pytorch-lightning (if applicable) to reduce the amount of boilerplate in your code

### Week 2

* [X] Write unit tests related to the data part of your code
* [X] Write unit tests related to model construction
* [X] Calculate the coverage.
* [X] Get some continuous integration running on the github repository
* [ ] (optional) Create a new project on `gcp` and invite all group members to it
* [ ] Create a data storage on `gcp` for you data
* [ ] Create a trigger workflow for automatically building your docker images
* [ ] Get your model training on `gcp`
* [X] Play around with distributed data loading
* [ ] (optional) Play around with distributed model training
* [ ] Play around with quantization and compilation for you trained models

### Week 3

* [ ] Deployed your model locally using TorchServe
* [ ] Checked how robust your model is towards data drifting
* [ ] Deployed your model using `gcp`
* [ ] Monitored the system of your deployed model
* [ ] Monitored the performance of your deployed model

### Additional

* [ ] Revisit your initial project description. Did the project turn out as you wanted?
* [ ] Make sure all group members have a understanding about all parts of the project
* [ ] Create a presentation explaining your project
* [ ] Uploaded all your code to github
* [ ] (extra) Implemented pre*commit hooks for your project repository
* [ ] (extra) Used Optuna to run hyperparameter optimization on your model
