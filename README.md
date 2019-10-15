# NLP-Bio-Tools #

NLP-Bio-Tools is a collection of applications for the text mining, Natural Language Processing (NLP) and Machine Learning (ML) classification of biomedical pdf documents.

The toolkit consists of three applications which are designed to be run with Docker. The accompanying Dockerfile for each application can be found in the application folder. The pre-built Docker images can be downloaded from [Docker Hub](https://hub.docker.com/u/annacprice). Docker can be run on [Windows](https://docs.docker.com/docker-for-windows/install/), [Mac](https://docs.docker.com/docker-for-mac/install/) and [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/).

## Workflow ##
<img height="400" src="https://github.com/annacprice/nlp-bio-tools/blob/master/workflow.png" />

The workflow for NLP-Bio-Tools is shown above. The first stage of the workflow is to extract and stem the text from the input pdfs (pdf2nlp). The workflow then diverges depending on whether you want to train a new ML model (mlpipe) or use a previously saved model to make predictions on new data (loadmodel). 

I.e. When building a new ML model:
```
pdf2nlp -> mlpipe
```
When evaluating a saved ML model on new data:
```
pdf2nlp -> loadmodel
```
The ML model built by mlpipe is a binary classification model. To build the model it requires a large training dataset which includes positive (belonging to a group) and negative (not belonging to a group) classes. The training dataset should be reflective of the "real world" data that the saved ML model is likely to encounter. Once the model is built, loadmodel can be used on new documents to predict which class they belong to.

A logistic regression model for the classification of articles into the HGMD is included in loadmodel/data/models. In this case, the model was built using articles belonging to the HGMD (the positive class), plus general PubMed articles and articles belonging to COSMIC (the negative class).

### pdf2nlp ###
pdf2nlp takes an input pdf and parses the text using pdfminer (note that pdfminer can only extract embedded text and cannot process scanned pdfs). The parsed text is then processed by a NLP pipeline that returns stemmed tokens, which the machine learning algorithm in mlpipe can fit to. 

A list of stopwords (pdf2nlp/data/biostopwords.txt) relevant to academic biomedical articles is used by pdf2nlp to remove common words which we don't want the model in mlpipe to fit to. A US-UK English dictionary (pdf2nlp/data/ustouk.txt) is used to convert all the text to UK English.

#### Setup: ####
* The input pdfs should be placed in pdf2nlp/data/papers
* The output txt files will be saved to pdf2nlp/data/output

To download the image from Docker Hub:
```
docker pull annacprice/pdf2nlp:1.1
```
To run the application in a Docker container:
```
cd pdf2nlp
docker run -v $(pwd)/data:/data --rm pdf2nlp
```

### mlpipe ###
mlpipe is used to build the machine learning model. The user can select which vectorizer and ML algorithm they wish to use to build the model. 

#### Setup: ####
* The postive and negative datasets should first be processed using pdf2nlp. The positive dataset should then be placed in mlpipe/data/text/positive and the negative dataset in mlpipe/data/text/negative
* A results file and ROC curve are saved to mlpipe/data/output, along with the saved vectorizer and model (vectorizer.pkl and model.pkl) which are used by loadmodel

The following vectorizers are available:
* [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
* [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

and the following ML algorithms:

* [KNeighborsClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) for k=5
* [SVC](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)
* [MultinomialNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.MultinomialNB.html)
* [BernoulliNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.BernoulliNB.html)
* [LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)

To download the image from Docker Hub:
```
docker pull annacprice/mlpipe:1.1
```

So for example to run the application in a Docker container with the TfidfVectorizer and the LogisticRegression model:
```
cd mlpipe
docker run -v $(pwd)/data:/data --rm mlpipe TfidfVectorizer LogisticRegression
```

### loadmodel ###
loadmodel takes the saved machine learning model from mlpipe and uses it to predict the class of new documents. 

#### Setup: ####
* The saved vectoriser and model should be placed in loadmodel/data/models 
* The txt files you wish to evaluate should be placed in loadmodel/data/text
* A results file is saved to loadmodel/data/output

To download the image from Docker Hub:
```
docker pull annacprice/loadmodel:1.1
```
To run the application in a Docker container:
```
cd loadmodel
docker run -v $(pwd)/data:/data --rm loadmodel vectorizer.pkl model.pkl
```
