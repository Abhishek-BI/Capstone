# -*- coding: utf-8 -*-
"""
Created on Fri Oct 09 11:30:46 2015

@author: Vaibhav
"""
import pandas as pd 
import numpy as np
import os
path = "C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\"
os.chdir(path+'train')    
import gzip
from sklearn.datasets import load_svmlight_file
from sklearn.mixture import GMM
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold

def maxCount(z):
    return z.value_counts().idxmax()

def pXoverC(X_train_two, y_train_two, X_test_two, y_test_two, X_val_two, y_val_two, X_train,  X_test,  X_val,  n_guass,i,j):
    s_train = np.array([None]*len(y_train))
    s_test = np.array([None]*len(y_test))
    s_val = np.array([None]*len(y_val))
    s_train_two = np.array([None]*len(y_train_two))
    s_test_two = np.array([None]*len(y_test_two))
    s_val_two = np.array([None]*len(y_val_two))
    for cls in np.unique(y_train_two):
        classifier = GMM(n_components=n_guass,covariance_type='full', init_params='wc', n_iter=20)
        classifier.fit(X_train_two[y_train_two==cls]) 
        print "trained classifier: " + str(cls)
        temp_train_two = classifier.score(X_train_two)
        s_train_two = np.vstack((s_train_two, temp_train_two))
        temp_test_two = classifier.score(X_test_two)
        s_test_two = np.vstack((s_test_two, temp_test_two))
        temp_val_two = classifier.score(X_val_two)
        s_val_two = np.vstack((s_val_two, temp_val_two))
        
        temp_train = classifier.score(X_train.todense())
        s_train = np.vstack((s_train, temp_train))
        temp_test = classifier.score(X_test.todense())
        s_test = np.vstack((s_test, temp_test))
        temp_val = classifier.score(X_val.todense())
        s_val = np.vstack((s_val, temp_val))
        
    x_train_two = pd.DataFrame(s_train_two.T,columns=['None',i,j])
    pred_train_two = x_train_two.idxmax(axis = 1)
    acc_train_two = np.mean(y_train_two==pred_train_two)   
    print 'acc_train: ' + str(acc_train_two)
    x_test_two = pd.DataFrame(s_test_two.T,columns=['None',i,j])
    pred_test_two = x_test_two.idxmax(axis = 1)
    acc_test_two = np.mean(y_test_two==pred_test_two)  
    print 'acc_test: ' + str(acc_test_two)    
    x_val_two = pd.DataFrame(s_val_two.T,columns=['None',i,j])
    pred_val_two = x_val_two.idxmax(axis = 1)
    acc_val_two = np.mean(y_val_two==pred_val_two) 
    print 'acc_val: ' + str(acc_val_two) 
    
    x_train = pd.DataFrame(s_train.T,columns=['None',i,j])
    pred_train = x_train.idxmax(axis = 1)
     
        
    x_test = pd.DataFrame(s_test.T,columns=['None',i,j])
    pred_test = x_test.idxmax(axis = 1)
        
        
    x_val = pd.DataFrame(s_val.T,columns=['None',i,j])
    pred_val = x_val.idxmax(axis = 1)
        
    return acc_train_two,acc_test_two,acc_val_two,pred_train,pred_test,pred_val

def featureSelection(X_train,X_test,X_val,y_train,log,tech,C):
    if (tech == 'VarTh'):
        sel = VarianceThreshold(threshold=0.01)
        X_train_new = sel.fit_transform(X_train.todense())
        X_test_new = sel.transform(X_test.todense())
        X_val_new = sel.transform(X_val.todense())
        if (log):
            X_train_new = np.log(X_train_new+1)
            X_test_new = np.log(X_test_new+1)
            X_val_new = np.log(X_val_new+1)
    
    if (tech == 'LinearSVC'):
        mod = LinearSVC(C=C, penalty="l1", dual=False)
        X_train_new = mod.fit_transform(X_train.todense(), y_train)
        X_test_new = mod.transform(X_test.todense())
        X_val_new = mod.transform(X_val.todense())
        if (log):
            X_train_new = np.log(X_train_new+1)
            X_test_new = np.log(X_test_new+1)
            X_val_new = np.log(X_val_new+1)
    return X_train_new, X_test_new , X_val_new

def plotAccuracy(fileName):
    data = pd.read_csv(fileName)
    plt.figure(figsize=(9,7))
    if (data.columns[0] != 'filename'):
        for i in range(1,len(data.columns)):
            plt.plot(data[data.columns[0]],data[data.columns[i]])
    else:
        x = range(len(data))
        plt.xticks(x,data[data.columns[0]],rotation='vertical')
        for i in range(1,len(data.columns)):
            plt.plot(x,data[data.columns[i]])
        
    plt.legend(data.columns[1:], loc='upper left')
    plt.xlabel(data.columns[0])
    plt.ylabel('Accuracy')
    plt.title('Accuracy plot for ' + fileName)
    plt.show()
#===================================Main =======================================
file ='vision_cuboids_histogram.txt.gz'
X_train, y_train = load_svmlight_file(gzip.open(path+"train\\"+file))
X_test, y_test = load_svmlight_file(gzip.open(path+"test\\"+file))
X_val, y_val = load_svmlight_file(gzip.open(path+"validation\\"+file))
X_train = X_train[y_train!=31]
X_test = X_test[y_test!=31]
X_val = X_val[y_val!=31]
y_train = y_train[y_train!=31]
y_test = y_test[y_test!=31]
y_val = y_val[y_val!=31]    

tech = 'LinearSVC'
C=0.5
X_train_new, X_test_new , X_val_new = featureSelection(X_train,X_test,X_val,y_train, log=True,tech = tech,C=C)

data_df = pd.DataFrame()
n_guass =5
pred_train_array = pd.DataFrame()
pred_test_array = pd.DataFrame()
pred_val_array = pd.DataFrame()
acc_mat = np.zeros(shape = (30,30))        

for i in range(1,31):
    for j in range(2,31):
        if (j>i):
            ClassPair = str(i)+':'+str(j)                      
            print 'Class Pair: ' +    ClassPair
            
            X_train_two = np.append(X_train_new()[y_train==i],X_train_new[y_train ==j],axis=0)
            y_train_two = np.append(y_train[y_train == i],y_train[y_train == j],axis=0)
        
            X_test_two = np.append(X_test_new[y_test==i],X_test_new[y_test ==j],axis=0)
            y_test_two = np.append(y_test[y_test == i],y_test[y_test == j],axis=0)
            
            X_val_two = np.append(X_val_new[y_val==i],X_val_new[y_val ==j],axis=0)
            y_val_two = np.append(y_val[y_val == i],y_val[y_val == j],axis=0)
            
            acc_train,acc_test,acc_val,pred_train,pred_test,pred_val = pXoverC(X_train_two, y_train_two, X_test_two, y_test_two, X_val_two, y_val_two,X_train,X_test,X_val, n_guass,i,j)
            
            acc_mat[i-1,j-1] = acc_train
            acc_mat[j-1,i-1] = acc_test
            temp = pd.DataFrame([[ClassPair,acc_train,acc_test,acc_val]])        
            data_df = data_df.append(temp,ignore_index =True)
            pred_train_array[ClassPair] = pred_train
            pred_test_array[ClassPair] = pred_test
            pred_val_array[ClassPair] = pred_val
            
acc_mat_pd = pd.DataFrame(acc_mat)
acc_mat_pd.to_csv("Accuracy Matrix_0910.csv",index=False)
data_df.columns = ['Class Pairs','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv('ClassPair_Acc_0910.csv',index=False)
train_vote = pred_train_array.apply(maxCount, axis=1)
test_vote = pred_test_array.apply(maxCount, axis=1)
val_vote = pred_val_array.apply(maxCount, axis=1)

vote_acc_train = np.mean(y_train==train_vote) 
vote_acc_test = np.mean(y_test==test_vote) 
vote_acc_val = np.mean(y_val==val_vote) 
data = pd.DataFrame([[file,vote_acc_train,vote_acc_test,vote_acc_val]])
data.columns = ['FileName','Vote train Accuracy','Vote test Accuracy','Vote validation Accuracy']
data.to_csv('Vote_Accurarcy_0910.csv',index=False)

plotAccuracy('ClassPair_Acc_0910.csv')    
plotAccuracy('Vote_Accurarcy_0910.csv')

    
    
    
    
    