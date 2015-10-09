# -*- coding: utf-8 -*-
"""
Created on Wed Oct 07 19:34:42 2015

@author: vaibhav
"""
os.chdir("F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\train")
path = "F:\\Analytics\\ISB Study\\Capstone\\dir_data\\dir_data\\"


file  = 'vision_cuboids_histogram.txt'
data_df = pd.DataFrame()
    
    train_post_array = []
    test_post_array = []
    val_post_array = []
    train_entropy_array = []
    test_entropy_array = []
    val_entropy_array = []
    fileType = featureFamily+'*.gz'
  
        #X_train, y_train, X_test, y_test,X_val, y_val = load_svmlight_files((gzip.open(path+"train\\"+file), gzip.open(path+"test\\"+file),gzip.open(path+"validation\\"+file)))    
        #X_train, y_train, X_test, y_test, X_val, y_val = load_svmlight_files(("train\\vision_cuboids_histogram.txt", "test\\vision_cuboids_histogram.txt","validation\\vision_cuboids_histogram.txt"))
        X_train, y_train = load_svmlight_file(file)
        X_train = X_train[y_train!=31]

        X_test, y_test = load_svmlight_file((path+"test\\"+file))
        X_test = X_test[y_test!=31]

        X_val, y_val = load_svmlight_file((path+"validation\\"+file))
        X_val = X_val[y_val!=31]
                        
        y_train = y_train[y_train!=31]
        y_test = y_test[y_test!=31]
        y_val = y_val[y_val!=31]
        
        
    s_train = np.array([None]*len(y_train))
    s_test = np.array([None]*len(y_test))
    s_val = np.array([None]*len(y_val))
    for cls in np.unique(y_train):
        classifier = GMM(n_components=n_guass,covariance_type='diag', init_params='wc', n_iter=50)
        classifier.fit(X_train.todense()[y_train==cls]) 
        print "trained classifier: " + str(cls)
        temp_train = classifier.predict_proba(X_train.todense())
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
        
        
s = classifier.score(X_train.todense())

from sklearn.utils.extmath import logsumexp
from sklearn import mixture
lpr = (mixture.log_multivariate_normal_density(X_train.todense(), classifier.means_, classifier.covars_, classifier.covariance_type) + np.log(classifier.weights_)) # probabilities of components
logprob = logsumexp(lpr, axis=1) # logsum to get probability of GMM
probs = np.exp(logprob) # 0 < probs < 1 

p = prior(y_train)
x = x_train + np.log(p)
x = x.astype(float)
z = x
r = logsumexp(d,axis = 1)
d = x.as_matrix()
x['logsum'] = r
z = np.exp(x.subtract(x['logsum'],axis=0).drop('logsum',1))



checkAccuracy(z,y_train)


train_prob,test_prob,val_prob = pXoverC(X_train.todense(), y_train, X_test.todense(), y_test, X_val.todense(), y_val, n_guass)


