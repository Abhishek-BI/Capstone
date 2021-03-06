# -*- coding: utf-8 -*-
"""
Created on Sat Oct 03 10:56:23 2015

@author: vaibhav
"""

from sklearn.mixture import GMM
import pandas as pd
import os
from sklearn.datasets import load_svmlight_files
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.lda import LDA
from sklearn.decomposition import PCA as sklearnPCA



def plot_pca(title):

    ax = plt.subplot(111)

    for label,marker,color in zip(
        range(28,31),('^', 's', 'o'),('yellow','black','red')):

        plt.scatter(x=X_pca[:,0][y_train == label],
                y=X_pca[:,1][y_train == label],
                marker=marker,
                color=color,
                alpha=0.5,
             
                )

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title(title)
    leg = plt.legend(loc='upper right', fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.tight_layout
    plt.grid()

    plt.show()
    
    
 
def plot_scikit_lda(X, title, mirror=1):

    ax = plt.subplot(111)
    for label,marker,color in zip(
        range(1,4),('^', 's', 'o'),('blue', 'red', 'green')):

        plt.scatter(x=X[:,0][y_train == label]*mirror,
                y=X[:,1][y_train == label],
                marker=marker,
                color=color,
                alpha=0.5,
                
                )

    plt.xlabel('LD1')
    plt.ylabel('LD2')
    plt.title(title)
    leg = plt.legend(loc='upper right', fancybox=True)
    #leg.get_frame().set_alpha(0.5)
    plt.title(title)

    # hide axis ticks
    plt.tick_params(axis="both", which="both", bottom="off", top="off",
            labelbottom="on", left="off", right="off", labelleft="on")

    # remove axis spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)

    plt.grid()
    plt.tight_layout
    plt.show()
    
    
    
os.chdir("F:\Analytics\ISB Study\Capstone\dir_data\dir_data")



X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt","validation\\vision_cuboids_histogram.txt"))
np.unique(y_train)

sklearn_lda = LDA(n_components=30)
X_lda_sklearn = sklearn_lda.fit_transform(X_train.todense(), y_train)
plot_scikit_lda(X_lda_sklearn, title='LDA vision_cuboids_histogram')
# PCA
sklearn_pca = sklearnPCA(n_components=30)
X_pca = sklearn_pca.fit_transform(X_train.todense())
plot_pca(title = 'PCA vision_cuboids_histogram')
#
X_ldapca_sklearn = sklearn_pca.fit_transform(X_lda_sklearn)
plot_scikit_lda(X_ldapca_sklearn, title='LDA+PCA LDA vision_cuboids_histogram', mirror=(-1))
