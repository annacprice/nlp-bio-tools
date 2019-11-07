# NLP-Bio-Tools #

NLP-Bio-Tools is a collection of applications for the text mining, Natural Language Processing (NLP) and Machine Learning (ML) classification of biomedical pdf documents. The toolkit consists of three applications (pdf2nlp, mlpipe and loadmodel) that are designed to be run with Docker using the directory structure found in this repository.

A quick start guide using a dummy dataset is provided below. For more detailed information on how to build your own machine learning model and how to use Docker, please consult the [wiki](https://github.com/annacprice/nlp-bio-tools/wiki).

Additionally, a logistic regression model for classification of articles into the HGMD can be found in the hgmd_model directory.

## Quick Start ##
The following quick start guide uses a dummy dataset to demonstrate how to build and use a machine learning model with NLP-Bio-Tools.

### Setup ###
The first step is to install Docker ([Windows](https://docs.docker.com/docker-for-windows/install/), [Mac](https://docs.docker.com/docker-for-mac/install/) and [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)). Once Docker is installed, the following commands can be used to pull the Docker images for each application from Docker Hub:
```
docker pull annacprice/pdf2nlp:1.1
docker pull annacprice/mlpipe:1.1
docker pull annacprice/loadmodel:1.1
```

We now need to download the latest release of the NLP-Bio-Tools repositiory found [here](https://github.com/annacprice/nlp-bio-tools/releases/tag/v1.1). On opening the zip file you should see the nlp-bio-tools-1.1 folder where the necessary directory structure and dummy dataset needed to run NLP-Bio-Tools can be found.

### Stage 1: pdf2nlp ###
pdf2nlp takes the input pdfs and passes them through a Natural Language Processing (NLP) pipeline. It outputs a txt file of tokens for each pdf which the ML algorithm in mlpipe can fit to.

The dummy pdf dataset to be processed can be found in pdf2nlp/data/papers and the expected txt output from the pdf2nlp container in pdf2nlp/data/output.

First navigate to the correct directory
```
cd nlp-bio-tools-1.1/pdf2nlp
```
And then run the container
```
docker run -v $(pwd)/data:/data --rm annacprice/pdf2nlp:1.1
```
This should reproduce the files in pdf2nlp/data/output 


### Stage 2: mlpipe ###
mlpipe builds a binary classification machine learning (ML) model. 

Some of the txt files produced by pdf2nlp have been placed in mlpipe/data/text to make a training set for mlpipe. They have been divided into positive and negative classes. The expected output of the mlpipe container (a txt results file, ROC curve, and the saved pkl files for the built ML model) can be found in mlpipe/data/output.

If using Mac OS you will need to remove .DS_Store files from the mlpipe/data/text/positive and mlpipe/data/text/negative directories before running the mlpipe container. I.e. when in the directory run
```
find . -name '.DS_Store' -type f -delete
```
First navigate to the correct directory
```
cd nlp-bio-tools-1.1/mlpipe
```
And then run the container with the CountVectorizer and KNeighborsClassifier algorithm
```
docker run -v $(pwd)/data:/data --rm annacprice/mlpipe:1.1 CountVectorizer KNeighborsClassifier
```
This should reproduce the files in mlpipe/data/output

More examples of vectorizers and ML algorithms that can be used with mlpipe are detailed in the [wiki](https://github.com/annacprice/nlp-bio-tools/wiki/How-to-build-your-own-machine-learning-model).

### Stage 3: loadmodel ###
The remaining txt files from pdf2nlp have been placed in loadmodel/data/text to act as a testing dataset for our ML model from mlpipe (the model.pkl and vectorizer.pkl files), which can be found in loadmodel/data/models. The expected output from the loadmodel container is a txt results file, and can be found in loadmodel/data/output. The second column in the results file indicates whether the original pdf has been classified into the negative (represented by a 0) or positive class (represented by a 1).

First navigate to the correct directory
```
cd nlp-bio-tools-1.1/loadmodel
``` 
And then run the container
```
docker run -v $(pwd)/data:/data --rm annacprice/loadmodel:1.1 vectorizer.pkl model.pkl
```

This should reproduce the MLresults.txt file in loadmodel/data/output

More information on how to evaluate the performance of your ML model can be found in the wiki [1](https://github.com/annacprice/nlp-bio-tools/wiki/How-to-build-your-own-machine-learning-model), [2](https://github.com/annacprice/nlp-bio-tools/wiki/How-to-use-a-saved-machine-learning-model).

## Acknowledgements ##
We acknowledge the support of the Supercomputing Wales project, which is part-funded by the European Regional Development Fund (ERDF) via Welsh Government.
