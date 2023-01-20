---
layout: default
nav_exclude: true
---

# Exam template for 02476 Machine Learning Operations

This is the report template for the exam. Please only remove the text formatted as with three dashes in front and behind
like:

```--- question 1 fill here ---```

where you instead should add your answers. Any other changes may have unwanted consequences when your report is auto
generated in the end of the course. For questions where you are asked to include images, start by adding the image to
the `figures` subfolder (please only use `.png`, `.jpg` or `.jpeg`) and then add the following code in your answer:

```markdown
![my_image](figures/<image>.<extension>)
```

In addition to this markdown file, we also provide the `report.py` scrcipt that provides two utility functions:

Running:

```bash
python report.py html
```

will generate an `.html` page of your report. After deadline for answering this template, we will autoscrape
everything in this `reports` folder and then use this utility to generate an `.html` page that will be your serve
as your final handin.

Running

```bash
python report.py check
```

will check your answers in this template against the constrains listed for each question e.g. is your answer too
short, too long, have you included an image when asked to.

For both functions to work it is important that you do not rename anything. The script have two dependencies that can
be installed with `pip install click markdown`.

## Group information

### Question 1
> **Enter the group number you signed up on <learn.inside.dtu.dk>**
>
> Answer:

60

### Question 2
> **Enter the study number for each member in the group**
>
> Example:
>
> *sXXXXXX, sXXXXXX, sXXXXXX*
>
> Answer:

s213422, s183685, s212834

### Question 3
> **What framework did you choose to work with and did it help you complete the project?**
>
> Answer length: 100-200 words.
>
> Example:
> *We used the third-party framework ... in our project. We used functionality ... and functionality ... from the*
> *package to do ... and ... in our project*.
>
> Answer:

We ended up using the TIMM framework, since our problem was one related to vision- We merely used the framework to load the needed models. To be honest, we didn't feel like it made much of a difference to use TIMM against something like torch.vision -  they seemed to be about the same thing. Although we recognize that TIMM has certain features we could have used, we ended up not doing so, as we felt they were not necessary for the scope of this project. None the less, we see the value in using third party frameworks, as these can help reduce the workload in setting up models, tests, schedulers, etc.

## Coding environment

> In the following section we are interested in learning more about you local development environment.

### Question 4

> **Explain how you managed dependencies in your project? Explain the process a new team member would have to go**
> **through to get an exact copy of your environment.**
>
> Answer length: 100-200 words
>
> Example:
> *We used ... for managing our dependencies. The list of dependencies was auto-generated using ... . To get a*
> *complete copy of our development enviroment, one would have to run the following commands*
>
> Answer:

We all worked in Anaconda, so, to ensure we could all have similar environments, we created two files: environment.txt, which we use to create a template conda environment, and requirements.txt, which includes all the needed python dependencies. If someone were to create an environment similar to the one we used for this project, one would only have to run "conda create --name <env> --file environment.txt", to create the conda env, and then run "pip install -r requirements.txt" to install the proper dependencies. Also, as the course went on, we found ourselves using new dependencies. To ensure we were all still working in the same env, we kept updating our requirements.txt file, and notify each other whenever there was a need for us to install new dependencies, which we did by running the aforementioned pip install command.

### Question 5

> **We expect that you initialized your project using the cookiecutter template. Explain the overall structure of your**
> **code. Did you fill out every folder or only a subset?**
>
> Answer length: 100-200 words
>
> Example:
> *From the cookiecutter template we have filled out the ... , ... and ... folder. We have removed the ... folder*
> *because we did not use any ... in our project. We have added an ... folder that contains ... for running our*
> *experiments.*
> Answer:

We extensively used the cookiecutter template (which we must add is a pretty nice template). We filed everything in subfolder /src, except /visualization, which we didn't use, so we removed it. We also removed /notebooks as we didn't use any Jupyter Notebooks. We also used the data folder to keep our data, but only the /processed subfolder. As we used DVC for our data, it was not needed to use the /raw folder, since the raw data is what we used for our model. We also added a few new scripts on the /src folder. Namely, we added a sweep_model.py script for using Weights and Biases functionalities, and we added a new subfolder to /src, /tests, which contains the scripts for the unittests.

### Question 6

> **Did you implement any rules for code quality and format? Additionally, explain with your own words why these**
> **concepts matters in larger projects.**
>
> Answer length: 50-100 words.
>
> Answer:

We followed the Pep8 for our code to ensure our project complies with the official python style. We used flake8 to see where we had code that wasn't complient with this style, and used black to automatically reformat the code into the required style. This is important in large projects, as it ensures all developers are following the same coding practices, making it easy for them to understand and integrate new features in code written by others. It's akin to being in a group where everyone is speaking a different language - sure, with Google Translate we would eventually understand eachother, but if we all speak English, it's probably easier for everyone to understand :)

## Version control

> In the following section we are interested in how version control was used in your project during development to
> corporate and increase the quality of your code.

### Question 7

> **How many tests did you implement?**
>
> Answer: 

We implemented 2 tests functions, one for the model testing if the output of the model has the right format, and one for the data testing different things such as the shape of images or the number of unique labels. We tried to implement a test for the training function as well, with less success. 

### Question 8

> **What is the total code coverage (in percentage) of your code? If you code had an code coverage of 100% (or close**
> **to), would you still trust it to be error free? Explain you reasoning.**
>
> **Answer length: 100-200 words.**
>
> Example:
> *The total code coverage of code is X%, which includes all our source code. We are far from 100% coverage of our **
> *code and even if we were then...*
>
> Answer: 

The total code coverage of the code is 70% according to the coverage report, including all the useful functions of the source code. If we tested the training function, we could reach a higher score. The data test function for example covers 98% of the data function with only one untested line. However, even if we reached 100% in total, the code wouldn't necessarily be error free as the coverage just checks how much of the code was run to get the test results, but doesn't evaluate if those tests are exhaustive enough or relevant. For example, if we test all the images that we get from the dataset class, by checking if they have the right shape, it will give good coverage while the image could in theory be filled with black pixels and not be properly loaded.

### Question 9

> **Did you workflow include using branches and pull requests? If yes, explain how. If not, explain how branches and**
> **pull request can help improve version control.**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of both branches and PRs in our project. In our group, each member had an branch that they worked on in*
> *addition to the main branch. To merge code we ...*
>
> Answer:

We ended up not using branches and pull requests. At the beggining, we did create an extra branch but quickly dropped it. Although we see the usefulness in large scale projects with many developers, we did use branches and pull requests, since we divided our tasks in such a way that there was barely any overlap in our coding. For example, one of us worked on the unit tests, so all the coding was done in the subfolder src/tests, while meanwhile another was working on the data modelling in src/features, and another on the model training in src/models. Since we were only three, and we mostly coded together in the same physical place, it was easy to keep track of eachothers progress, and ensure there was no overlap. Of course this wouldn't have been possible if we were a large MLOps teams of 20 or 30 people - there, without a doubt using branches can help, as the tasks are divided into their own self contained goals, and developers can work on different things in the same code at the same time, without breaking the whole pipeline - here pull requests will facilitate the integration of the code into one.

### Question 10

> **Did you use DVC for managing data in your project? If yes, then how did it improve your project to have version**
> **control of your data. If no, explain a case where it would be beneficial to have version control of your data.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did make use of DVC in the following way: ... . In the end it helped us in ... for controlling ... part of our*
> *pipeline*
>
> Answer:

We used DVC mainly for a matter of simplyfying the data management process. In reality, it was not necessary to use DVC - our dataset was "static", in the sense that we didn't expect it to change throughout the project: we didn't antecipate any changes or extra data being added. None the less, despite us not really using the data version control benefits, which help you when you have data that can go through changes, we found it very helpful to remotely store our data and being able to pull it to a new machine with relative simplicity. In the case that our dataset was, for example, too small, DVC would be helpful in keeping track of new data added through time, and keeping old versions of the data easily accessible. 

### Question 11

> **Discuss you continues integration setup. What kind of CI are you running (unittesting, linting, etc.)? Do you test**
> **multiple operating systems, python version etc. Do you make use of caching? Feel free to insert a link to one of**
> **your github actions workflow.**
>
> Answer length: 200-300 words.
>
> Example:
> *We have organized our CI into 3 separate files: one for doing ..., one for running ... testing and one for running*
> *... . In particular for our ..., we used ... .An example of a triggered workflow can be seen here: <weblink>*
>
> Answer:

We have organized our CI into 2 separate YAML files. The first one is used for testing and runs all the unit tests using the pytest library. The second one is used for linting the code. It first runs isort to sort the imports in each file of the source code, and then it runs flake8 on the source code to check if there are some formatting errors. For the unit testing, the process is run on 6 different configurations, with 3 different OS (Ubuntu, Windows, and macOS) and 2 different python versions (3.8 and 3.9). It also makes use of caching to accelerate the installation of pip libraries. However, the unit tests for data are skipped on GitHub actions as the data files cannot be accessed. The data folder is too heavy to be loaded and running "dvc pull" in the workflow file leads to a memory error. We thought about creating another bucket containing a subset of the data to reduce its size, but we did not have time to implement it. The linting workflow only runs once on Ubuntu and Python 3.8, and without cache, as we are only installing 2 pip libraries. Here are links to GitHub actions for each workflow file: [unit-testing](https://github.com/arturfabricio/project-mlops/actions/runs/3960456261), [linting](https://github.com/arturfabricio/project-mlops/actions/runs/3960644271).

## Running code and tracking experiments

> In the following section we are interested in learning more about the experimental setup for running your code and
> especially the reproducibility of your experiments.

### Question 12

> **How did you configure experiments? Did you make use of config files? Explain with coding examples of how you would**
> **run a experiment.**
>
> Answer length: 50-100 words.
>
> Example:
> *We used a simple argparser, that worked in the following way: python my_script.py --lr 1e-3 --batch_size 25*
>
> Answer:

We initially used click to add values to our variables, such as the epochs, learning rate, model, etc. from the terminal, very much similar to the example above. In the very end, we ended up not doing this (hence why click options are commented in the train model script), due to us using Weights and Biases. A more thorough explanation of this can be found in the next question.

### Question 13

> **Reproducibility of experiments are important. Related to the last question, how did you secure that no information**
> **is lost when running experiments and that your experiments are reproducible?**
>
> Answer length: 100-200 words.
>
> Example:
> *We made use of config files. Whenever an experiment is run the following happens: ... . To reproduce an experiment*
> *one would have to do ...*
>
> Answer:

We didn't use any config files, as we had a slightly different approach to training our model, and ensuring reproducibility. As we used Weights and Biases, what we did was run the sweep_model.py script, which performed a hyperparameter sweep for our model. This helped us figure out which hyperparameters were most adequate to ensure a higher validation accuracy. From this, we then simply inputted the best hyperparameter set on the script for the training model. Furthermore, to keep track of each model trained, we named the resulting pytorch model dict based on the used learning rate, epochs and batch size.

### Question 14

> **Upload 1 to 3 screenshots that show the experiments that you have done in W&B (or another experiment tracking**
> **service of your choice). This may include loss graphs, logged images, hyperparameter sweeps etc. You can take**
> **inspiration from [this figure](figures/wandb.png). Explain what metrics you are tracking and why they are**
> **important.**
>
> Answer length: 200-300 words + 1 to 3 screenshots.
>
> Example:
> *As seen in the first image when have tracked ... and ... which both inform us about ... in our experiments.*
> *As seen in the second image we are also tracking ... and ...*
>
> Answer:

The major use we gave to Weights and Biases (wandb) was on the aspect of performing an hyperparameter sweep, as seen [in this figure](figures/our_sweep.png). We aimed at using wandb to determine what were the optimal set of hyperparameters, namely, number of epochs, learning rate, and batch size, that would maximized the obtained validation accuracy obtained in the training process. It should be noted that as you can see in the aforementioned picture, all the sets of hyperparameters maximized the validation accuracy to 100%. This is because the model here is training on a very small set of images (400), so the model overfitted greatly. This is because we didn't have time as well as GCP credits to train the model with the full 101000 images of the datatset. None the less, this exemplifies an implementation of the hyperparameter sweep functionality of wandb in our code. Furthermore, we also logged several other parameters, such as graphs for the validation and accuracy loss and accuracy, which are also relevant indicatiors of the training process. This can be observed [here](figures/our_loss_acc.png). Furthermore, one more logging capibility of wandb is to be able to record system parameters, as seen [here](figures/our_system.png). These give us an insight into the GPU performance (if one is being used) in the form of Memory Allocated, Temperature, etc. 

### Question 15

> **Docker is an important tool for creating containerized applications. Explain how you used docker in your**
> **experiments? Include how you would run your docker images and include a link to one of your docker files.**
>
> Answer length: 100-200 words.
>
> Example:
> *For our project we developed several images: one for training, inference and deployment. For example to run the*
> *training docker image: `docker run trainer:latest lr=1e-3 batch_size=64`. Link to docker file: <weblink>*
>
> Answer:

The docker images were created for both training and inference. Since our dataset is as big as it is we had to exclude the day from the docker image which needs to be downloaded seperatly. For the training it simply just made it run the train script where the parameters are set within the python file. We figured that it would make the most sence to use weights and biases for now to calculate the best hyper parameters and then train it. The inference dockerfile simply just prints whatever food type the model think it is given an image.png which is given insde the image. For future implmentations it would ne nice to have some sort of automated pipeline which is able to train from the images given the latest best hyper parameters from Weight and Biases. Also being able to give inputs to the docker file so you could test the inference on different images. 

### Question 16

> **When running into bugs while trying to run your experiments, how did you perform debugging? Additionally, did you**
> **try to profile your code or do you think it is already perfect?**
>
> Answer length: 100-200 words.
>
> Example:
> *Debugging method was dependent on group member. Some just used ... and others used ... . We did a single profiling*
> *run of our main code at some point that showed ...*
>
> Answer:

When it cames to debugging, we all used the good old method of printing. To be fair we did encounter many bugs in the code, so it all went quite smoothly on this end. In regards to profiling the code, we actually did use the PyTorch profiler, implemented in a way very similar to what we did in the exercises. You can see the implementations in src/models/train_model.py. We did not spend much time looking into this, we did a simple run, where we were able to see that the conv layers and the backward pass were the operations that were taking the longest in average, which is what we expected.

## Working in the cloud

> In the following section we would like to know more about your experience when developing in the cloud.

### Question 17

> **List all the GCP services that you made use of in your project and shortly explain what each service does?**
>
> Answer length: 50-200 words.
>
> Example:
> *We used the following two services: Engine and Bucket. Engine is used for... and Bucket is used for...*
>
> Answer:

Cloud storage buckets: This is used to store relevant data for your project this will most likely include models and training data. Compute Engine: Compute engines have multiple purposes, it can be used as a working enviorment for developing purposes but also to run multiple VM's in parallel. The VM's also have the option to run on either CPU or GPU which have different advantages. Cloud build: Cloud build can be used to connect a git repository to thus creating updated version of images as the code updates. These images are then stores inside the container registry. Monitoring: Even though monitoring is a fairly new concept within machine learning it is relevant to keep track of essential features when both training and deploying the model. 

### Question 18

> **The backbone of GCP is the Compute engine. Explained how you made use of this service and what type of VMs**
> **you used?**
>
> Answer length: 50-100 words.
>
> Example:
> *We used the compute engine to run our ... . We used instances with the following hardware: ... and we started the*
> *using a custom container: ...*
>
> Answer:

The compute engine was made used to actually train the model. Combining the VM's with tmux its possible to let the training run over night and in with parralel sessions. If we had more money we would have setup multiple VM's using GPUs to train the model faster. So for this project and to illustrate that we understand the purpose we simply used the standard hardware which is: 1-2 vCPU and 4 GB memory. The VM was also created using a standard pytorch image.  

### Question 19

> **Insert 1-2 images of your GCP bucket, such that we can see what data you have stored in it.**
> **You can take inspiration from [this figure](figures/bucket.png).**
>
> Answer:

The bukcets were mainly for handling the data. The data was configured using DVC for version control. If we got the cloud functions to work, then the buckets would also be in charge of storing relavant models and other data for the function to run. For this project we used Compute engines to train but it would also have been an option to use Vertex AI. For this we would need to upload the data inside a bucket not as DVC format. Technically our dataset would not need DVC since its fixed. [Bucket image](figures/our_bucket.png) [Bucket data image](figures/our_bucket_data.png) 

### Question 20

> **Upload one image of your GCP container registry, such that we can see the different images that you have stored.**
> **You can take inspiration from [this figure](figures/registry.png).**
>
> Answer:

For the container registry we only made one type of image which was in charge of handling the training. This was mainly done to get the vertex AI up and running, which unfortunatly did not happen. The training image still has its purpose since it can be run on a local machine or used to setup a desired VM. The images for the training is inside the folder called "mlops-final". For future implementation it would be a good idea to also make images in charge of the testing and inference. [this figure](figures/our_registry.png)

### Question 21

> **Upload one image of your GCP cloud build history, so we can see the history of the images that have been build in**
> **your project. You can take inspiration from [this figure](figures/build.png).**
>
> Answer:

The GCP cloud build history can be seen here: [this figure](figures/our_build.png)

### Question 22

> **Did you manage to deploy your model, either in locally or cloud? If not, describe why. If yes, describe how and**
> **preferably how you invoke your deployed service?**
>
> Answer length: 100-200 words.
>
> Example:
> *For deployment we wrapped our model into application using ... . We first tried locally serving the model, which*
> *worked. Afterwards we deployed it in the cloud, using ... . To invoke the service an user would call*
> *`curl -X POST -F "file=@file.json"<weburl>`*
>
> Answer:

We manges to deploy the model but only locally. The API which was created simply uses an image as an input and then using our trained model will output the food type which it think it is It was intented to get the model deployed to the cloud, but it showed to be a fairly difficult task for us. A common error we got was "Function failed on loading user code. This is likely due to a bug in the user code". Which is not very specific and fairly difficult to debug when the deployment works locally. To get acces to our service the user would have to write either: curl -X POST -F "image_file=@path\to\file.jpg" http://localhost:8000/predict/. Run the python script called "apitest.py". Which uses the requst libary to access the API. or simply use the following link: http://localhost:8000/docs, which usesa a simple FastAPI user interface to test the API. 

### Question 23

> **Did you manage to implement monitoring of your deployed model? If yes, explain how it works. If not, explain how**
> **monitoring would help the longevity of your application.**
>
> Answer length: 100-200 words.
>
> Example:
> *We did not manage to implement monitoring. We would like to have monitoring implemented such that over time we could*
> *measure ... and ... that would inform us about this ... behaviour of our application.*
>
> Answer:

We did implement monitoring in the form of an alert that would notify us if the number of bytes in ingested log entries went over 1. This was not particularly useful, and was done mainly to test the implementation of a telemtry alert in our project. In another context, we believe we could have set up monitoring on other more meaningful aspects. These could be, for example, an alert on the sent bytes of our GCS Bucket where our data is stored, hinting at the fact someone might be downloading, or perhaps, set up an alert on the Uptime of a VM instace - this could be particularly useful in the case of GCS, due to the use of credits to run virtual machines. 

### Question 24

> **How many credits did you end up using during the project and what service was most expensive?**
>
> Answer length: 25-100 words.
>
> Example:
> *Group member 1 used ..., Group member 2 used ..., in total ... credits was spend during development. The service*
> *costing the most was ... due to ...*
>
> Answer:

Lucas used: 21.97 credits, Artur used: 6.81 credits and César used: ??. The credits were spent during development mainly due to the testing and training inside the VM's. 

## Overall discussion of project

> In the following section we would like you to think about the general structure of your project.

### Question 25

> **Include a figure that describes the overall architecture of your system and what services that you make use of.**
> **You can take inspiration from [this figure](figures/overview.png). Additionally in your own words, explain the**
> **overall steps in figure.**
>
> Answer length: 200-400 words
>
> Example:
>
> *The starting point of the diagram is our local setup, where we integrated ... and ... and ... into our code.*
> *Whenever we commit code and puch to github, it auto triggers ... and ... . From there the diagram shows ...*
>
> Answer:

An overview of our overall architecture can be seen [here](figures/our_overview.png). We can start by explaining that we extensivily use GCP in this project: Data Version Control (DVC) is performed on data stored in a Google Cloud Bucket; the training is performed in virtual machines (VM's) in GCP, and deployment is also performed using Cloud Functions. In terms of version control, we use GitHub. Everytime we commit, unitests are auto triggered using GitHub actions, ensuring our code is compliant with whatever parameters tested in the unittests. Furthermore, we also take advantage of Weights and Biases to perform hyperparameter sweeps to determine what are the most adequate hyperparameters. Weights and Biases is also used as a logging tool, which saves the results of the training process, in variables such as test/val accuracy, test/val loss, etc., as well as the overall use of hardware elements for the training of the model. From a User perspective, one can easily pull the latest version of the code, and use the latest Docker image available. Furthermore, deployment is relatively simple as well, as we have cloud functions that allow to deploy the code (for now this is not on the cloud but merely local ADD EXTRA EXPLANATiON HERE).

### Question 26

> **Discuss the overall struggles of the project. Where did you spend most time and what did you do to overcome these**
> **challenges?**
>
> Answer length: 200-400 words.
>
> Example:
> *The biggest challenges in the project was using ... tool to do ... . The reason for this was ...*
>
> Answer:

One of the challenges we initially faced in the project was setting up DVC, since we initially set everything up on Google Drive, which was a mess as each of us had to log into one Google account to have the permissions to access the data. Furthermore, we initially couldn't even get DVC to work in Google Drive at all. Despite this, we solved the issue by migrating our data to a GCS Bucket, and making the bucket public, which made it easy for everyone to access it. We initially also struggled a bit with implementing the Weights and Biases features on the code (just as we had on the exercises). We then figured that the order where in which we call each wandb command is crucial for the features to work, so we eventually fixed the issue. Docker, GCP and deployment were all somewhat new areas for all of us which made some part quite a big challenge, and also led to some unfinished parts. Getting the cloud functions to work was definitly the biggest challenge, never having done any deployment before it made it a bit hard to just get it done locally. Even though there are alot of logs to look at for the cloud functoin for debugging it would often give a very unspecific error which is hard to debug without experience. It was also a bit of mess understanding the pipeline for the docker building and pushing. We had problems getting multiple triggers to work at the same time which lead to only one trigger active for the training and the the inference not. 

### Question 27

> **State the individual contributions of each team member. This is required information from DTU, because we need to**
> **make sure all members contributed actively to the project**
>
> Answer length: 50-200 words.
>
> Example:
> *Student sXXXXXX was in charge of developing of setting up the initial cookie cutter project and developing of the*
> *docker containers for training our applications.*
> *Student sXXXXXX was in charge of training our models in the cloud and deploying them afterwards.*
> *All members contributed to code by...*
>
> Answer:

Student s214322 (Artur): worked mainly on the data handling - making the data ready for training, as well as setting DVC. Also worked on getting the model training, and implemented wandb features.
Student s212834 (César): worked mainly on getting the model training, and was responsible for the design and implementation of unit tests and continous integration on the GitHub repo.
Student s183685 (Lucas): worked mainly on setting up docker environments, as well as setting up the features on GCP, and working on the API and deployment of our model.
In terms of the code, it is very hard to precise how much each contributed. We all worked on the project together, and often we'd correct each others code, and thus we believe we all contributed equally.