# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:34:06 2015

@author: Vaibhav
"""

import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import NuSVC
os.chdir("C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data")



from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import load_svmlight_file
from sklearn import svm

mem = Memory("./mycache")

@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]



#text_game_lda_1000
X, y = load_svmlight_file("train\\text_game_lda_1000.txt")

Xtest, ytest = load_svmlight_file("test\\text_game_lda_1000.txt")

text_clf = Pipeline([ ('clf', SGDClassifier(loss='log', penalty='l1',alpha=1e-6, n_iter=10, random_state=88)),])

text_clf = text_clf.fit(X, y)
prob = pd.DataFrame(text_clf.predict_proba(Xtest))
predicted = text_clf.predict(Xtest)
print "Validation Model Accuracy %f" % np.mean(predicted == ytest) # 92.82%

clf = svm.SVC()
clf.fit(X.todense(), y)
predicted = clf.predict(Xtest)
print "Validation Model Accuracy %f" % np.mean(predicted == ytest) # 92.82%


from sklearn.svm import LinearSVC
mod = LinearSVC()
mod.fit(X,y)
predicted = mod.predict(Xtest)
print "Validation Model Accuracy %f" % np.mean(predicted == ytest) # 92.82%
mod.pred
#text_description_unigrams
X_train, y_train, X_test, y_test = load_svmlight_files(("train\\text_description_unigrams.txt", "test\\text_description_unigrams.txt"))

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-6, n_iter=5, random_state=88)),])

text_clf = text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
print "Validation Model Accuracy %f" % np.mean(predicted == y_test) # 68.21%

#text_tag_unigrams
X_train, y_train, X_test, y_test = load_svmlight_files(("train\\text_tag_unigrams.txt", "test\\text_tag_unigrams.txt"))

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l1',alpha=1e-6, n_iter=5, random_state=88)),])

text_clf = text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
print "Validation Model Accuracy %f" % np.mean(predicted == y_test) # 92.05%


#audio_mfcc
X_train, y_train, X_test, y_test = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt"))

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-6, n_iter=5, random_state=88)),])

text_clf = text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
print "Validation Model Accuracy %f" % np.mean(predicted == y_test) 

#==============================================================================#

file = "text_game_lda_1000.txt.gz"

#==========================================Text SVM First Level ===================================================

import glob, os
import pandas as pd
import gzip
os.chdir("C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\train")
path = "C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\"
data_df = pd.DataFrame()

train_prob_array = []
test_prob_array = []
val_prob_array = []

for file in glob.glob("audio*.gz"):
    print(file)
    X_train, y_train, X_test, y_test,X_val, y_val = load_svmlight_files((gzip.open(path+"train\\"+file), gzip.open(path+"test\\"+file),gzip.open(path+"validation\\"+file)))    
        
    X_train = X_train[y_train!=31]
    X_test = X_test[y_test!=31]
    X_val = X_val[y_val!=31]
    y_train = y_train[y_train!=31]
    y_test = y_test[y_test!=31]
    y_val = y_val[y_val!=31]
    
    svmClf = Pipeline([ ('clf', SGDClassifier(loss='log', penalty='l1',alpha=1e-6, n_iter=10, random_state=88)),])
    svmClf = svmClf.fit(X_train, y_train)
    
    predicted_train = svmClf.predict(X_train)
    train_acc = np.mean(predicted_train == y_train)     
    print "Train Model Accuracy %f" % train_acc    
    train_prob = pd.DataFrame(svmClf.predict_proba(X_train))
    
    predicted_test = svmClf.predict(X_test)
    test_acc = np.mean(predicted_test == y_test)        
    print "Test Model Accuracy %f" % test_acc
    test_prob = pd.DataFrame(svmClf.predict_proba(X_test))    
    
    predicted_val = svmClf.predict(X_val)
    val_acc = np.mean(predicted_val == y_val)     
    print "Validation Model Accuracy %f" % val_acc
    val_prob = pd.DataFrame(svmClf.predict_proba(X_val))    
    
    temp = pd.DataFrame([[file,train_acc,test_acc,val_acc]])        
    data_df = data_df.append(temp,ignore_index =True)
    train_prob_array.append(train_prob)
    test_prob_array.append(test_prob)
    val_prob_array.append(val_prob)

data_df.columns = ['filename','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv("SVM_Accuracy_audio.csv")

#==============================================================================
from sklearn import decomposition
import matplotlib.pyplot as plt

pca = decomposition.PCA()

pca.fit(X_train.toarray())

plt.figure(1, figsize=(10, 10))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_, linewidth=2)
plt.axis('tight')   
plt.xlabel('n_components')
plt.ylabel('explained_variance_')


import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import SGDClassifier, Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier

heldout = [0.95, 0.90, 0.75, 0.50, 0.01]
rounds = 20



classifiers = [
    ("SGD", SGDClassifier()),
    ("ASGD", SGDClassifier(average=True)),
    ("Perceptron", Perceptron()),
    ("Passive-Aggressive I", PassiveAggressiveClassifier(loss='hinge',
                                                         C=1.0)),
    ("Passive-Aggressive II", PassiveAggressiveClassifier(loss='squared_hinge',
                                                          C=1.0)),
]

for name, clf in classifiers:
    rng = np.random.RandomState(42)
    yy = []
    for i in heldout:
        yy_ = []
        for r in range(rounds):
            X_train, X_test, y_train, y_test = \
                train_test_split(X, y, test_size=i, random_state=rng)
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            yy_.append(1 - np.mean(y_pred == y_test))
        yy.append(np.mean(yy_))
    plt.plot(xx, yy, label=name)

from sklearn.neighbors.nearest_centroid import NearestCentroid
clf = NearestCentroid()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
            
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
logreg = SVC(kernel='linear', C=1.0, probability=True,random_state=0)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
np.mean(y_pred == y_test)








