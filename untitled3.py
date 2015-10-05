# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 10:37:29 2015

@author: vaibhav
"""

import numpy as np

import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.mixture import GMM
import pandas as pd
import os
from sklearn.datasets import load_svmlight_files
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC

from sklearn.feature_selection import VarianceThreshold


def pXoverC(X_train, y_train, X_test, y_test, X_val, y_val, n_guass):
    s_train = np.array([None]*len(y_train))
    s_test = np.array([None]*len(y_test))
    s_val = np.array([None]*len(y_val))
    for cls in np.unique(y_train):
        classifier = GMM(n_components=n_guass,covariance_type='full', init_params='wc', n_iter=50)
        classifier.fit(X_train[y_train==cls]) 
        print "trained classifier: " + str(cls)
        temp_train = classifier.predict_proba(X_train)
        s_train = np.vstack((s_train, (classifier.weights_*temp_train).sum(axis =1)))
        temp_test = classifier.predict_proba(X_test)
        s_test = np.vstack((s_test, (classifier.weights_*temp_test).sum(axis =1)))
        temp_val = classifier.predict_proba(X_val)
        s_val = np.vstack((s_val, (classifier.weights_*temp_val).sum(axis =1)))
        
        x_train = pd.DataFrame(s_train.T)
        x_test = pd.DataFrame(s_test.T)
        x_val = pd.DataFrame(s_val.T)
    x_train = x_train.drop(x_train.columns[[0]],axis = 1)
    x_test = x_test.drop(x_test.columns[[0]],axis = 1)
    x_val = x_val.drop(x_val.columns[[0]],axis = 1)
    return x_train, x_test, x_val 


def entropy(z):
    z = z.astype(float)
    z_log = np.log(z)
    z_entropy = -z*z_log
    return z_entropy.sum(1)

def prior(y):
    prior = []
    for cls in np.unique(y):
        p = list(y).count(cls)/float(len(y))
        prior.append(round(p,5))
    return prior


def posterior(pXoverC, prior):
    x = pXoverC*prior
    x['sum'] = x.sum(axis=1)
    z = x.div(x['sum'],axis = 0).drop('sum',1)
    return z

def mixGuass(X_train, y_train, X_test, y_test, X_val, y_val,components):
    data_df = pd.DataFrame()
    for n_comp in components:
        print "Number of Guassians per class: " + str(n_comp)
        s_train = np.array([None]*len(y_train))
        s_test = np.array([None]*len(y_test))
        s_val = np.array([None]*len(y_val))
        for cls in np.unique(y_train):
            classifier = GMM(n_components=n_comp,covariance_type='full', init_params='wc', n_iter=50)
            classifier.fit(X_train[y_train==cls]) 
            print "trained classifier: " + str(cls)
            temp_train = classifier.score(X_train)
            s_train = np.vstack((s_train,temp_train))
            temp_test = classifier.score(X_test)
            s_test = np.vstack((s_test,temp_test))
            temp_val = classifier.score(X_val)
            s_val = np.vstack((s_val,temp_val))
       
        x_train = pd.DataFrame(s_train.T)
        pred_train = x_train.idxmax(axis = 1)
        acc_train = np.mean(y_train==pred_train)   
        
        x_test = pd.DataFrame(s_test.T)
        pred_test = x_test.idxmax(axis = 1)
        acc_test = np.mean(y_test==pred_test)  
        
        x_val = pd.DataFrame(s_val.T)
        pred_val = x_val.idxmax(axis = 1)
        acc_val = np.mean(y_val==pred_val)  
        
        temp = pd.DataFrame([[n_comp,acc_train,acc_test,acc_val]])        
        data_df = data_df.append(temp,ignore_index =True)
    
    data_df.columns = ['NumOfGuassians','train_Accuracy','test_Accuracy','validation_Accuracy']
    return data_df


def combiner(posteriorsArray,entropyArray,alpha):
    zeros_data = np.zeros(shape = posteriorsArray[0].shape)
    y = pd.DataFrame(zeros_data,columns = range(1,31))
    for i in range(len(posteriorsArray)):
        x = posteriorsArray[i].multiply(np.power(1-entropyArray[i],alpha),axis = 0)
        y = y + x
    
    zeros_data = np.zeros(shape = entropyArray[0].shape)
    z = pd.DataFrame(zeros_data,columns = range(1))
    for i in range(len(entropyArray)):
        z = z + pd.DataFrame(np.power(1-entropyArray[i],alpha))

    return y.divide(z.ix[:,0],axis = 0)    
    

def checkAccuracy(X,Y):
     pred = X.idxmax(axis = 1)
     acc = np.mean(Y==pred)  
     print "Accuracy: " + str(acc)
     return acc, confusion_matrix(Y, pred)



#======================================================================================================================    
#os.chdir("F:\Analytics\ISB Study\Capstone\dir_data\dir_data")
os.chdir("C:\Users\Vaibhav\Desktop\dir_data\dir_data")

X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt","validation\\vision_cuboids_histogram.txt"))
np.unique(y_train)
#========================= Removing Class 31 =============================================================
X_train = X_train[y_train!=31]
X_test = X_test[y_test!=31]
X_val = X_val[y_val!=31]
y_train = y_train[y_train!=31]
y_test = y_test[y_test!=31]
y_val = y_val[y_val!=31]
#========================= Feature Selection using Variance Thresold =============================================================
sel = VarianceThreshold(threshold=0.0001)

X_train_new = sel.fit_transform(X_train.todense())
X_test_new = sel.transform(X_test.todense())
X_val_new = sel.transform(X_val.todense())
X_train_new = np.log(X_train_new+1)
X_test_new = np.log(X_test_new+1)
X_val_new = np.log(X_val_new+1)

#========================= Mixture of Guassian ============================================================
#components = [5]
#data = mixGuass(X_train_new, y_train, X_test_new, y_test, X_val_new, y_val,components)
n_guass = 2
p1,p2,p3 = pXoverC(X_train_new, y_train, X_test_new, y_test, X_val_new, y_val, n_guass)
#========================= Calculating Prior, Posterior and Entropy ============================================================
x = prior(y_train)
z = posterior(p1,x)
z_entropy = entropy(z)

#z_pow = np.power(1-z_entropy,2)
#z_mult = z.multiply(z_pow,0)
posteriorsArray = [p1,p11]
entropyArray = [z_entropy,z_entropy1]
alpha=1

d = combiner(posteriorsArray,entropyArray,alpha)

trainAcc, c_mat = checkAccuracy(d,y_train)
     






















