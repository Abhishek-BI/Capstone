# -*- coding: utf-8 -*-
"""
Created on Sun Oct 04 10:37:29 2015

@author: vaibhav
"""

import numpy as np
import glob, os
import gzip
from sklearn.mixture import GMM
import pandas as pd
from sklearn.datasets import load_svmlight_files
from sklearn.datasets import load_svmlight_file
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from sklearn.utils.extmath import logsumexp
# ========================================== Define Functions ================================================
def pXoverC(X_train, y_train, X_test, y_test, X_val, y_val, n_guass):
    s_train = np.array([None]*len(y_train))
    s_test = np.array([None]*len(y_test))
    s_val = np.array([None]*len(y_val))
    for cls in np.unique(y_train):
        classifier = GMM(n_components=n_guass,covariance_type='full', init_params='wc', n_iter=50)
        classifier.fit(X_train[y_train==cls]) 
        print "trained classifier: " + str(cls)
        temp_train = classifier.score(X_train)
        s_train = np.vstack((s_train, temp_train))
        temp_test = classifier.score(X_test)
        s_test = np.vstack((s_test, temp_test))
        temp_val = classifier.score(X_val)
        s_val = np.vstack((s_val, temp_val))
        
    x_train = pd.DataFrame(s_train.T)
    x_test = pd.DataFrame(s_test.T)
    x_val = pd.DataFrame(s_val.T)
    x_train = x_train.drop(x_train.columns[[0]],axis = 1)
    x_test = x_test.drop(x_test.columns[[0]],axis = 1)
    x_val = x_val.drop(x_val.columns[[0]],axis = 1)
    return x_train, x_test, x_val 


def entropy(z,nClass):
    z = z.astype(float)
    z_log = np.log(z)/np.log(nClass)
    z_entropy = -z*z_log
    return z_entropy.sum(axis = 1)

def prior(y):
    prior = []
    for cls in np.unique(y):
        p = list(y).count(cls)/float(len(y))
        prior.append(round(p,5))
    return prior


def posterior(pXoverC, prior):
    #x = pXoverC*prior
    #x['sum'] = x.sum(axis=1)
    #z = x.div(x['sum'],axis = 0).drop('sum',1)
    x = pXoverC + np.log(prior)
    x = x.astype(float)
    x['logsum'] = logsumexp(x.as_matrix(),axis = 1)
    z = np.exp(x.subtract(x['logsum'],axis=0).drop('logsum',1))
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
    y = pd.DataFrame(zeros_data,columns = range(1,posteriorsArray[0].shape[1]+1))
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

  
def pCoverX(featureFamily,n_guass,tech,C,nClass):
    #os.chdir("F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\train")
    #path = "F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\"
    path = "C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\"
    os.chdir(path+'train')    
    
    data_df = pd.DataFrame()
    nClass = 30
    train_post_array = []
    test_post_array = []
    val_post_array = []
    train_entropy_array = []
    test_entropy_array = []
    val_entropy_array = []
    fileType = featureFamily+'*.gz'
    for file in glob.glob(fileType):
        print file
        #X_train, y_train, X_test, y_test,X_val, y_val = load_svmlight_files((gzip.open(path+"train\\"+file), gzip.open(path+"test\\"+file),gzip.open(path+"validation\\"+file)))    
        #X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt","validation\\vision_cuboids_histogram.txt"))
        X_train, y_train = load_svmlight_file(gzip.open(path+"train\\"+file))
        X_train = X_train[y_train!=31]
        #X_train = X_train[y_train <=2]

        X_test, y_test = load_svmlight_file(gzip.open(path+"test\\"+file))
        X_test = X_test[y_test!=31]
        #X_test = X_test[y_test <=2]

        X_val, y_val = load_svmlight_file(gzip.open(path+"validation\\"+file))
        X_val = X_val[y_val!=31]
        #X_val = X_val[y_val <=2]        
                
        y_train = y_train[y_train!=31]
        y_test = y_test[y_test!=31]
        y_val = y_val[y_val!=31]
        #y_train = y_train[y_train <=2]
        #y_test = y_test[y_test <= 2]
        #y_val = y_val[y_val <=2]
        print "File Read"
    #========================= Feature Selection using Variance Thresold =============================================================
        X_train_new, X_test_new , X_val_new = featureSelection(X_train,X_test,X_val,y_train, log=True,tech = tech,C=C)
    #========================= Mixture of Guassian ============================================================
        train_prob,test_prob,val_prob = pXoverC(X_train_new, y_train, X_test_new, y_test, X_val_new, y_val, n_guass)
    #========================= Calculating Prior, Posterior and Entropy ============================================================
        prr = prior(y_train)
        train_post = posterior(train_prob,prr)
        train_entropy = entropy(train_post,nClass)
        
        train_post_array.append(train_post)
        train_entropy_array.append(train_entropy)
    
        test_post = posterior(test_prob,prr)
        test_entropy = entropy(test_post,nClass)
    
        test_post_array.append(test_post)
        test_entropy_array.append(test_entropy)
        
        val_post = posterior(val_prob,prr)
        val_entropy = entropy(val_post,nClass)
    
        val_post_array.append(val_post)
        val_entropy_array.append(val_entropy)
        
        train_acc,c_mat = checkAccuracy(train_post,y_train)
        test_acc,c_mat = checkAccuracy(test_post,y_test)
        val_acc,c_mat = checkAccuracy(val_post,y_val)
        temp = pd.DataFrame([[file,train_acc,test_acc,val_acc]])        
        data_df = data_df.append(temp,ignore_index =True)
        
    return train_post_array,test_post_array,val_post_array,train_entropy_array,test_entropy_array,val_entropy_array,data_df

def textpCoverX():
    #os.chdir("F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\train")
    #path = "F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\"
    path = "C:\\Users\\Vaibhav\\Desktop\\dir_data\\dir_data\\"
    os.chdir(path+'train')
    
    data_df = pd.DataFrame()
    
    train_post_array = []
    test_post_array = []
    val_post_array = []
    train_entropy_array = []
    test_entropy_array = []
    val_entropy_array = []
    
    for file in glob.glob("text*.gz"):
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
        train_post = pd.DataFrame(svmClf.predict_proba(X_train))
        
        predicted_test = svmClf.predict(X_test)
        test_acc = np.mean(predicted_test == y_test)        
        print "Test Model Accuracy %f" % test_acc
        test_post = pd.DataFrame(svmClf.predict_proba(X_test))    
        
        predicted_val = svmClf.predict(X_val)
        val_acc = np.mean(predicted_val == y_val)     
        print "Validation Model Accuracy %f" % val_acc
        val_post = pd.DataFrame(svmClf.predict_proba(X_val))    
        
        
        train_entropy = entropy(train_post)
        
        train_post_array.append(train_post)
        train_entropy_array.append(train_entropy)
    
        test_entropy = entropy(test_post)
    
        test_post_array.append(test_post)
        test_entropy_array.append(test_entropy)
        
        val_entropy = entropy(val_post)
    
        val_post_array.append(val_post)
        val_entropy_array.append(val_entropy)
        
        temp = pd.DataFrame([[file,train_acc,test_acc,val_acc]])        
        data_df = data_df.append(temp,ignore_index =True)
        
    return train_post_array,test_post_array,val_post_array,train_entropy_array,test_entropy_array,val_entropy_array,data_df


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

#=============================================== Main =================================================================

#os.chdir("F:\Analytics\ISB Study\Capstone\dir_data\dir_data")
os.chdir("C:\Users\Vaibhav\Desktop\dir_data\dir_data")
X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_hist_motion_estimate.txt", "test\\vision_hist_motion_estimate.txt","validation\\vision_hist_motion_estimate.txt"))
y_train = y_train[y_train!=31]
y_test = y_test[y_test!=31]
y_val = y_val[y_val!=31]
#y_train = y_train[y_train <=2]
#y_test = y_test[y_test<=2]
#y_val = y_val[y_val<=2]
#================ First Level of Fusion - Audio ===============================
n_guass =5
nClass = 30 
train_post_array,test_post_array,val_post_array,train_entropy_array,test_entropy_array,val_entropy_array,data_df = pCoverX('audio',n_guass,tech = 'LinearSVC',C= 0.5,nClass=30)
data_df.columns = ['filename','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv('Audio_preComb_Acc0801.csv',index=False)

audioComb1Acc = pd.DataFrame()
for alpha in [1,2,3,4,5]:
    comb1_audio_train = combiner(train_post_array,train_entropy_array,alpha)
    comb1_audio_test = combiner(test_post_array,test_entropy_array,alpha)
    comb1_audio_val = combiner(val_post_array,val_entropy_array,alpha)

    audioTrainAcc, c_mat = checkAccuracy(comb1_audio_train,y_train)
    audioTestAcc, c_mat = checkAccuracy(comb1_audio_test,y_test)
    audioValAcc, c_mat = checkAccuracy(comb1_audio_val,y_val)
     
    temp = pd.DataFrame([[alpha,audioTrainAcc,audioTestAcc,audioValAcc]]) 
    audioComb1Acc = audioComb1Acc.append(temp)
    
audioComb1Acc.columns = ['alpha','train Accuracy','test Accuracy','validation Accuracy']
audioComb1Acc.to_csv("Audio_combiner1_Acc0801.csv",index=False)
audio_train_entropy = entropy(comb1_audio_train,nClass)
audio_test_entropy = entropy(comb1_audio_test,nClass)
audio_val_entropy = entropy(comb1_audio_val,nClass)

#================ First Level of Fusion - Video ===============================
n_guass = 5
train_post_array,test_post_array,val_post_array,train_entropy_array,test_entropy_array,val_entropy_array,data_df = pCoverX('vision',n_guass,tech = 'LinearSVC',C=0.5,nClass=30)
data_df.columns = ['filename','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv('Vision_preComb_Acc0801.csv',index=False)
visionComb1Acc  = pd.DataFrame()
for alpha in [1,2,3,4,5]:
    comb1_vision_train = combiner(train_post_array,train_entropy_array,alpha)
    comb1_vision_test = combiner(test_post_array,test_entropy_array,alpha)
    comb1_vision_val = combiner(val_post_array,val_entropy_array,alpha)
    
    visionTrainAcc, c_mat = checkAccuracy(comb1_vision_train,y_train)
    visionTestAcc, c_mat = checkAccuracy(comb1_vision_test,y_test)
    visionValAcc, c_mat = checkAccuracy(comb1_vision_val,y_val)
         
    temp = pd.DataFrame([[alpha,visionTrainAcc,visionTestAcc,visionValAcc]])
    visionComb1Acc = visionComb1Acc.append(temp,ignore_index =True)    
    
visionComb1Acc.columns = ['alpha','train Accuracy','test Accuracy','validation Accuracy']
visionComb1Acc.to_csv("Vision_combiner1_Acc0801.csv",index=False)
vision_train_entropy = entropy(comb1_vision_train,nClass)
vision_test_entropy = entropy(comb1_vision_test,nClass)
vision_val_entropy = entropy(comb1_vision_val,nClass)

#================ First Level of Fusion - Text ================================
train_post_array,test_post_array,val_post_array,train_entropy_array,test_entropy_array,val_entropy_array,data_df = textpCoverX()
data_df.columns = ['filename','train Accuracy','test Accuracy','validation Accuracy']
data_df.to_csv("Text_preComb_Acc.csv",index=False)

textComb1Acc = pd.DataFrame()
for alpha in [1,2,3,4,5]:
    comb1_text_train = combiner(train_post_array,train_entropy_array,alpha)
    comb1_text_test = combiner(test_post_array,test_entropy_array,alpha)
    comb1_text_val = combiner(val_post_array,val_entropy_array,alpha)
    
    textTrainAcc, c_mat = checkAccuracy(comb1_text_train,y_train)
    textTestAcc, c_mat = checkAccuracy(comb1_text_test,y_test)
    textValAcc, c_mat = checkAccuracy(comb1_text_val,y_val)
         
    temp = pd.DataFrame([[alpha,textTrainAcc,textTestAcc,textValAcc]])
    textComb1Acc = textComb1Acc.append(temp,ignore_index =True)    
    
textComb1Acc.columns = ['alpha','train Accuracy','test Accuracy','validation Accuracy']
textComb1Acc.to_csv("Text_combiner1_Acc.csv",index=False)
text_train_entropy = entropy(comb1_text_train)
text_test_entropy = entropy(comb1_text_test)
text_val_entropy = entropy(comb1_text_val)

#=======================Second Level of Audio and Video fusion ================================================================
train_post_array = [comb1_audio_train,comb1_vision_train]
train_entropy_array = [audio_train_entropy,vision_train_entropy]

test_post_array = [comb1_audio_test,comb1_vision_test]
test_entropy_array = [audio_test_entropy,vision_test_entropy]

val_post_array = [comb1_audio_val,comb1_vision_val]
val_entropy_array = [audio_val_entropy,vision_val_entropy]

Comb2Acc = pd.DataFrame()
for beta in [1,2,3,4,5]:
    comb2_train = combiner(train_post_array,train_entropy_array,beta)
    comb2_test = combiner(test_post_array,test_entropy_array,beta)
    comb2_val = combiner(val_post_array,val_entropy_array,beta)
    
    finalTrainAcc, c_mat = checkAccuracy(comb2_train,y_train)
    finalTestAcc, c_mat = checkAccuracy(comb2_test,y_test)
    finalValAcc, c_mat = checkAccuracy(comb2_val,y_val)
    
    temp = pd.DataFrame([[beta,finalTrainAcc,finalTestAcc,finalValAcc]])
    Comb2Acc = Comb2Acc.append(temp,ignore_index =True)

Comb2Acc.columns = ['beta','train Accuracy','test Accuracy','validation Accuracy']
Comb2Acc.to_csv("Final_combiner2_Acc0801.csv",index=False)

#=============================================================================================
#================ Second Level of Fusion - Audio, Video & Text ===============================
train_post_array = [comb1_audio_train,comb1_vision_train,comb1_text_train]
train_entropy_array = [audio_train_entropy,vision_train_entropy,text_train_entropy]

test_post_array = [comb1_audio_test,comb1_vision_test,comb1_text_test]
test_entropy_array = [audio_test_entropy,vision_test_entropy,text_test_entropy]

val_post_array = [comb1_audio_val,comb1_vision_val,comb1_text_val]
val_entropy_array = [audio_val_entropy,vision_val_entropy,text_val_entropy]

Comb2Acc = pd.DataFrame()
for beta in [1,2,3,4,5]:
    comb2_train = combiner(train_post_array,train_entropy_array,beta)
    comb2_test = combiner(test_post_array,test_entropy_array,beta)
    comb2_val = combiner(val_post_array,val_entropy_array,beta)
    
    finalTrainAcc, c_mat = checkAccuracy(comb2_train,y_train)
    finalTestAcc, c_mat = checkAccuracy(comb2_test,y_test)
    finalValAcc, c_mat = checkAccuracy(comb2_val,y_val)
    
    temp = pd.DataFrame([[beta,finalTrainAcc,finalTestAcc,finalValAcc]])
    Comb2Acc = Comb2Acc.append(temp,ignore_index =True)

Comb2Acc.columns = ['beta','train Accuracy','test Accuracy','validation Accuracy']
Comb2Acc.to_csv("Final_combiner2_Acc07.csv",index=False)

#==============================Plots ==========================================
plotAccuracy('Audio_preComb_Acc0801.csv')
plotAccuracy('Audio_combiner1_Acc0801.csv')
plotAccuracy('Vision_preComb_Acc0801.csv')
plotAccuracy('Vision_combiner1_Acc0801.csv')
plotAccuracy('Final_combiner2_Acc0801.csv')







