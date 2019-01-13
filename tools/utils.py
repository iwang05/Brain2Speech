import os.path
import numpy as np
import pickle
import sys
import pylab as plt
from sklearn import datasets


def plot_2dclassifier(model, X, y):
    """plots the decision boundary of the model and the scatterpoints
       of the target values 'y'.

    Assumptions
    -----------
    y : it should contain two classes: '0' and '1'

    Parameters
    ----------
    model : the trained model which has the predict function

    X : the N by 2 feature array

    y : the N element vector corresponding to the target values

    """
    x1 = X[:, 0]
    x2 = X[:, 1]

    x1_min, x1_max = int(x1.min()) - 1, int(x1.max()) + 1
    x2_min, x2_max = int(x2.min()) - 1, int(x2.max()) + 1

    x1_line =  np.arange(x1_min, x1_max)
    x2_line =  np.arange(x2_min, x2_max)

    x1_mesh, x2_mesh = np.meshgrid(x1_line, x2_line)

    mesh_data = np.c_[x1_mesh.ravel(), x2_mesh.ravel()]

    y_pred = model["predict"](model, mesh_data)
    y_pred = np.reshape(y_pred, x1_mesh.shape)

    index_0 = np.where(y==0)[0]
    index_1 = np.where(y==1)[0]

    #print(x1[index])

    plt.xlim([x1_mesh.min(), x1_mesh.max()])
    plt.ylim([x2_mesh.min(), x2_mesh.max()])

    plt.contourf(x1_mesh, x2_mesh, y_pred,
                cmap=plt.cm.RdBu_r, label="decision boundary",
                alpha=0.6)

    #plt.scatter(x1[y==0], x2[y==0], color="b", label="class 0")
    #plt.scatter(x1[y==1], x2[y==1], color="r", label="class 1")
    plt.scatter(x1[index_0], x2[index_0], color="b", label="class 0")
    plt.scatter(x1[index_1], x2[index_1], color="r", label="class 1")

    plt.legend()
    plt.title("Model outputs '0' for red region\n"
              "Model outputs '1' for blue region")


def mode(y):
    """Computes the element with the maximum count

    Parameters
    ----------
    y : an input numpy array

    Returns
    -------
    y_mode :
        Returns the element with the maximum count
    """
    if y.ndim > 1:
        y = y.ravel()
    N = y.shape[0]

    if N == 0:
        return -1

    keys = np.unique(y)

    counts = {}
    for k in keys:
        counts[k] = 0

    # Compute counts for each element
    for n in range(N):
        counts[y[n]] += 1

    y_mode = keys[0]
    highest = counts[y_mode]

    # Find highest count key
    for k in keys:
        if counts[k] > highest:
            y_mode = k
            highest = counts[k]

    return y_mode


def euclidean_dist_squared(X, Xtest):
    """Computes the Euclidean distance between rows of 'X' and rows of 'Xtest'

    Parameters
    ----------
    X : an N by D numpy array
    Xtest: an T by D numpy array

    Returns: an array of size N by T containing the pairwise squared Euclidean distances. 

    """

    # add extra dimensions so that the function still works for X and/or Xtest are 1-D arrays. 
    if X.ndim == 1:
        X = X[None]
    if Xtest.ndim == 1:
        Xtest = Xtest[None]

    return np.sum(X**2, axis=1)[:,None] + np.sum(Xtest**2, axis=1)[None] - 2 * np.dot(X,Xtest.T)
