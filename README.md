# NLP-Bio-Tools #

NLP-Bio-Tools is a collection of programs for the text mining, Natural Language Processing (NLP) and Machine Learning (ML) classification of biomedical pdf documents.

The toolkit consists of three applications which are designed to be run with Docker. The accompanying Dockerfile for each application can be found in the application folder. The pre-built Docker images can also be downloaded from Docker Hub. Docker can be run on [Windows](https://docs.docker.com/docker-for-windows/install/), [Mac](https://docs.docker.com/docker-for-mac/install/) and [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

## Workflow ##
<img height="400" src="https://github.com/annacprice/nlp-bio-tools/blob/master/workflow.png" />

### pdf2nlp ###
pdf2nlp takes an input pdf and parses the text using pdfminer. The parsed text is then processed by a NLP pipeline that returns stemmed tokens which the machine learning algorithm in mlpipe can fit to.

The Docker image for pdf2nlp can be found here. Alternatively, it can be built from the included Dockerfile like so:
```
docker build -t pdf2nlp .
```
To run the application in a Docker container:
```
docker run -v $(pwd)/data:/data --rm pdf2nlp
```

### mlpipe ###
mlpipe can be used to build a machine learning model 

The Docker image for mlpipe can be found here. Alternatively, it can be built from the included Dockerfile like so:

To run the application in a Docker container:

### loadmodel ###
loadmodel takes the saved machine learning model from mlpipe and uses it to evaluate new data. An example model for classification of articles into the HGMD is included in the application.

The Docker image for loadmodel can be found here. Alternatively, it can be built from the included Dockerfile like so:
```
docker build -t classifier .
```
To run the application in a Docker container:
```
docker run -v $(pwd)/data:/data --rm classifier vectorizer.pkl model.pkl
```
