# -*- coding: utf-8 -*-
"""
Created on Mon Sep 14 14:34:06 2015

@author: Vaibhav
"""

import os
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline

os.chdir("C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data")



from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import load_svmlight_file

mem = Memory("./mycache")

@mem.cache
def get_data(filename):
    data = load_svmlight_file(filename)
    return data[0], data[1]



#text_game_lda_1000
X, y = load_svmlight_file("train\\text_game_lda_1000.txt")

Xtest, ytest = load_svmlight_file("test\\text_game_lda_1000.txt")

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l1',alpha=1e-6, n_iter=10, random_state=88)),])

text_clf = text_clf.fit(X, y)

predicted = text_clf.predict(Xtest)
print "Validation Model Accuracy %f" % np.mean(predicted == ytest) # 92.82%


#text_description_unigrams
X_train, y_train, X_test, y_test = load_svmlight_files(("train\\text_description_unigrams.txt", "test\\text_description_unigrams.txt"))

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l1',alpha=1e-6, n_iter=5, random_state=88)),])

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
X_train, y_train, X_test, y_test = load_svmlight_files(("train\\audio_mfcc.txt", "test\\audio_mfcc.txt"))

text_clf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-6, n_iter=5, random_state=88)),])

text_clf = text_clf.fit(X_train, y_train)

predicted = text_clf.predict(X_test)
print "Validation Model Accuracy %f" % np.mean(predicted == y_test) 

#==============================================================================#

file = "text_game_lda_1000.txt.gz"



import glob, os
import pandas as pd
import gzip
os.chdir("C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\train")
path = "C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\"
data_df = pd.DataFrame()

for file in glob.glob("vision*.gz"):
    print(file)
    X_train, y_train, X_test, y_test,X_val, y_val = load_svmlight_files((gzip.open(path+"train\\"+file), gzip.open(path+"test\\"+file),gzip.open(path+"validation\\"+file)))    

    svmClf = Pipeline([ ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-6, n_iter=5, random_state=88)),])
    svmClf = svmClf.fit(X_train, y_train)
    
    predicted_train = svmClf.predict(X_train)
    train_acc = np.mean(predicted_train == y_train)     
    print "Train Model Accuracy %f" % train_acc    
    
    
    predicted_test = svmClf.predict(X_test)
    test_acc = np.mean(predicted_test == y_test)        
    print "Test Model Accuracy %f" % test_acc
    
    predicted_val = svmClf.predict(X_val)
    val_acc = np.mean(predicted_val == y_val)     
    print "Validation Model Accuracy %f" % val_acc
    
    temp = pd.DataFrame([[file,train_acc,test_acc,val_acc]])        
    data_df = data_df.append(temp,ignore_index =True)
    

data_df.columns = ['filename','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv("SVM_Accuracy_vision.csv")






    















