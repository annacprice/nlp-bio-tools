#!/usr/bin/env python2
#----------------------------------------------------------------------
# Author: Anna Price

# This script is part of the bioNLPtools software tool.

# It takes the input txt files created using pdf2nlp and uses
# them to build a binary classification machine learning model.
# Note 0 is the negative class and 1 is the positive.

# This script is designed to be used with the accompanying Dockerfile.
# The txt files from pdf2nlp should be placed in either the positive
# or negative folder in data/text.
# The resultant ML model is saved to data/output.

# To build the Docker image from current directory:
# docker build -t mlpipe .

# To run the program in the Docker container from current directory:
# docker run -v $(pwd)/data:/data --rm mlpipe vectorizer model
#----------------------------------------------------------------------
import sys
import os
import sklearn.datasets
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import roc_curve, auc, matthews_corrcoef
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from scipy import interp
import numpy as np
import matplotlib as mpl
if os.environ.get('DISPLAY', '') == '':
    mpl.use('Agg')
import matplotlib.pyplot as plt
from sklearn.externals import joblib


# set path for the input txt files for training the model
path='/data/text/'
categories_train=['negative', 'positive']


def build_model(vec, model):
    # select vectorizer
    if vec =='CountVectorizer':
        vec = CountVectorizer(input='string', max_df=0.8, min_df=0.02, \
                              max_features=750, ngram_range=(1,2), \
                              )
    elif vec=='TfidfVectorizer':
        vec = TfidfVectorizer(input='string', max_df=0.8, min_df=0.02, \
                              max_features=750, ngram_range=(1,2), \
                              )
    else:
        print('Error: vectoriser not recognised')
        exit()

    # select ML model
    if model=='LogisticRegression':
        clf = LogisticRegression()
    elif model=='KNeighborsClassifier':
        clf = KNeighborsClassifier(5)
    elif model=='SVC':
        clf = SVC(kernel='linear', C=0.025, probability=True)
    elif model=='MultinomialNB':
        clf = MultinomialNB()
    elif model=='BernoulliNB':
        clf = BernoulliNB(binarize=0.0)
    else:
        print('Error: ML model not recognised')
        exit()

    # load the training dataset
    model_train = sklearn.datasets.load_files(container_path=path, \
                                              categories=categories_train, \
                                              random_state=42)
    # apply vectorizer to the training dataset
    X_train_counts = vec.fit_transform(model_train.data)
    # save vectorizer
    joblib.dump(vec, '/data/output/vectorizer.pkl')
    # fit the training dataset to the model
    clf.fit(X_train_counts, model_train.target)
    # evaluate model on the training dataset
    predict = clf.predict(X_train_counts)
    # save the model
    joblib.dump(clf, '/data/output/model.pkl')

    # define 10-fold cross-validation of the dataset
    k_fold = StratifiedKFold(n_splits=10)
    kscore = cross_val_score(clf, X_train_counts, model_train.target, cv=k_fold, n_jobs=1)

    # save results to txt file
    with open('/data/output/mlpipe.txt' ,'a') as write_file:
        write_file.write('------------------------------------------------------------------\n')
        write_file.write('CLASSIFIER:\n')
        write_file.write(str(clf) + '\n')
        write_file.write('------------------------------------------------------------------\n')
        write_file.write('TRAINING:\n')
        write_file.write('10-FOLD CLASSIFICATION:\n')
        write_file.write('Accuracy on training set:\n')
        write_file.write(str(clf.score(X_train_counts, model_train.target)) + '\n')
        write_file.write('Classification report:\n')
        write_file.write(str(metrics.classification_report(model_train.target, predict)) + '\n')
        write_file.write('Confusion matrix:\n')
        write_file.write(str(metrics.confusion_matrix(model_train.target, predict)) + '\n')
        write_file.write('Matthews coefficient:\n')
        write_file.write(str(matthews_corrcoef(model_train.target, predict)) + '\n')

    # plot ROC curves
    ROC_plot(X_train_counts, model_train, k_fold, clf)


def ROC_plot(X_train_counts, model_train, k_fold, clf):
    # function to plot ROC curve
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    X = X_train_counts
    y = model_train.target
    
    # produce ROC curve for k-fold classification of training set
    i = 0
    for train, test in k_fold.split(X, y):
        probas_ = clf.fit(X[train], y[train]).predict_proba(X[test])
        # Compute ROC curve and area the curve
        fpr, tpr, thresholds = roc_curve(y[test], probas_[:,1], pos_label=1)
        tprs.append(interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        plt.plot(fpr, tpr, lw=1, alpha=0.3,
                 label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
             
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='g',
             label='AUC=0.5', alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, color='b',
         label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
         lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.05, 1.05])
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(str(clf), fontsize=12)
    plt.legend(loc='lower right')
    plt.savefig('/data/output/roc.png')


if __name__ == "__main__":
    build_model(*sys.argv[1:])
