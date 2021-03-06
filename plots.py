# -*- coding: utf-8 -*-
"""
Created on Sat Oct 03 12:21:05 2015

@author: vaibhav
"""
import numpy as np
import matplotlib.pyplot as plt

import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(3, 3)
file = 'text_tag_unigrams.txt.gz'
X_train, y_train  = load_svmlight_file(gzip.open(path+"train\\"+file) )

    plt.figure(figsize=(9,7))
    fig = plt.figure()    
    plt.title(file)
    #ax = fig.add_subplot(gs[0, 0])
    numBins = 20
    plt.hist(X_train.todense()[:,0],numBins,range=[0.00, 0.05],color='green',alpha=0.8)
    plt.show()
    
    
    x = 0.005
    fig = plt.figure(figsize=(9,7))    
    plt.title(file)
    ax = fig.add_subplot(gs[0, 0])
    numBins = 20
    ax.hist(X_train.todense()[:,0],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[0, 1])
    numBins = 20
    ax.hist(X_train.todense()[:,1],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[0, 2])
    numBins = 20
    ax.hist(X_train.todense()[:,2],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 0])
    numBins = 20
    ax.hist(X_train.todense()[:,3],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 1])
    numBins = 20
    ax.hist(X_train.todense()[:,4],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 2])
    numBins = 20
    ax.hist(X_train.todense()[:,5],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 0])
    numBins = 20
    ax.hist(X_train.todense()[:,6],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 1])
    numBins = 20
    ax.hist(X_train.todense()[:,7],numBins,range=[0.00, x],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 2])
    numBins = 20
    ax.hist(X_train.todense()[:,8],numBins,range=[0.00, x],color='green',alpha=0.8)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    fig = plt.figure()
    ax = fig.add_subplot(gs[0, 0])
    numBins = 20
    ax.hist(r[:,0],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[0, 1])
    numBins = 20
    ax.hist(r[:,1],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[0, 2])
    numBins = 20
    ax.hist(r[:,2],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 0])
    numBins = 20
    ax.hist(r[:,3],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 1])
    numBins = 20
    ax.hist(r[:,4],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[1, 2])
    numBins = 20
    ax.hist(r[:,5],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 0])
    numBins = 20
    ax.hist(r[:,6],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 1])
    numBins = 20
    ax.hist(r[:,7],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    ax = fig.add_subplot(gs[2, 2])
    numBins = 20
    ax.hist(r[:,8],numBins,range=[0.00, 0.02],color='green',alpha=0.8)
    plt.show()
    
  