# -*- coding: utf-8 -*-
"""
Created on Sat Oct 03 01:19:48 2015

@author: vaibhav
"""


import numpy as np


from sklearn.mixture import GMM
import pandas as pd

from sklearn.datasets import load_svmlight_files


X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt","validation\\vision_cuboids_histogram.txt"))
np.unique(y_train)


data_df = pd.DataFrame()
for n_comp in [1,2,3,4]:
    print "Number of Guassians per class: " + str(n_comp)
    s_train = np.array([None]*len(y_train))
    s_test = np.array([None]*len(y_test))
    s_val = np.array([None]*len(y_val))
    for cls in np.unique(y_train):
        classifier = GMM(n_components=n_comp,covariance_type='full', init_params='wc', n_iter=20)
        classifier.fit(X_train.todense()[y_train==cls]) 
        print "trained classifier: " + str(cls)
        temp_train = classifier.score(X_train.todense())
        s_train = np.vstack((s_train,temp_train))
        temp_test = classifier.score(X_test.todense())
        s_test = np.vstack((s_test,temp_test))
        temp_val = classifier.score(X_val.todense())
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

data_df.to_csv("vision_cuboids_histogram_gmm_acc.csv")



classifier0 = GMM(n_components=3,covariance_type='full', init_params='wc', n_iter=20)
classifier0.fit(X_train[y_train==0])   
classifier1 = GMM(n_components=3,covariance_type='full', init_params='wc', n_iter=20)
classifier1.fit(X_train[y_train==1])  
classifier2 = GMM(n_components=3,covariance_type='full', init_params='wc', n_iter=20)
classifier2.fit(X_train[y_train==2])  

z0 = classifier0.predict_proba(X_test)
z1 = classifier1.predict_proba(X_test)
z2 = classifier2.predict_proba(X_test)
s0 = classifier0.score(X_test)
s1 = classifier1.score(X_test)
s2 = classifier2.score(X_test)

n = np.vstack((a,s0,s1,s2))
x = pd.DataFrame(n.T)
p = x.idxmax(axis = 1)
np.mean(y_test==p)