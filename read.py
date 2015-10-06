# -*- coding: utf-8 -*-
"""
Created on Mon Oct 05 13:10:37 2015

@author: Vaibhav
"""

X_train1, y_train1, X_test1, y_test1, X_val1, y_val1 = load_svmlight_files(("train\\vision_hist_motion_estimate.txt", "test\\vision_hist_motion_estimate.txt","validation\\vision_hist_motion_estimate.txt"))
np.unique(y_train1)
#========================= Removing Class 31 =============================================================
X_train1 = X_train1[y_train1!=31]
X_test1 = X_test1[y_test1!=31]
X_val1 = X_val1[y_val1!=31]
y_train1 = y_train1[y_train1!=31]
y_test1 = y_test1[y_test1!=31]
y_val1 = y_val1[y_val1!=31]
#========================= Feature Selection using Variance Thresold =============================================================
sel = VarianceThreshold(threshold=0.0001)

X_train_new1 = sel.fit_transform(X_train1.todense())
X_test_new1 = sel.transform(X_test1.todense())
X_val_new1 = sel.transform(X_val1.todense())
X_train_new1 = np.log(X_train_new1+1)
X_test_new1 = np.log(X_test_new1+1)
X_val_new1 = np.log(X_val_new1+1)

#========================= Mixture of Guassian ============================================================

n_guass = 2
p11,p12,p13 = pXoverC(X_train_new1, y_train1, X_test_new1, y_test1, X_val_new1, y_val1, n_guass)

x1 = prior(y_train1)
z1 = posterior(p11,x1)
z_entropy1 = entropy(z1)

file ='audio_sai_boxes.txt.gz'
X_train, y_train = load_svmlight_file(gzip.open(path+"train\\"+file))
X_train = X_train[y_train!=31]

X_test, y_test = load_svmlight_file(gzip.open(path+"test\\"+file))
X_test = X_test[y_test!=31]

X_val, y_val = load_svmlight_file(gzip.open(path+"validation\\"+file))
X_val = X_test[y_val!=31]