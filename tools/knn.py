"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np
from scipy import stats
import utils

def fit(X, y, k):
    """
    Parameters
    ----------
    X : an N by D numpy array
    y : an N by 1 numpy array of integers in {1,2,3,...,c}
    k : the k in k-NN
    """
    # Just memorize the training dataset
    model = dict()
    model['X'] = X
    model['y'] = y
    model['k'] = k
    model['predict'] = predict
    return model

def predict(model, Xtest):
    """ YOUR CODE HERE """
    X=model['X']
    y=model['y']
    k=model['k']
    distance=utils.euclidean_dist_squared(X,Xtest)
    d_i=np.argsort(distance,axis=0) #array of indexes
    k_label=np.zeros([1,k])
    n,t=distance.shape
    yhat=np.zeros([1,t])
    for i in range(t):
        for j in range(k):
            k_label[0,j]=y[d_i[j,i]]
        yhat[0,i]=utils.mode(k_label[0])
    return yhat
    #yhat is a row

    #raise NotImplementedError
