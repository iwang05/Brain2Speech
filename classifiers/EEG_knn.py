import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import knn
import utils
import EEG_feature_extraction
import EEG_load


# main - train&test
#dataset = EEG_load.load_data("s16",20)

dataset = EEG_feature_extraction.generate_feature_data("s16",20)

X = dataset['X_train']
y = dataset['Y_train']
Xtest = dataset['X_test']
ytest = dataset['Y_test']

k=[1,3,10]
for i in range(3):
    model=knn.fit(X,y,k[i])
    y_pred=knn.predict(model,X)
    train_error=np.mean(y_pred.flatten() != y)
    print("The current training error is: %r" % train_error)

    y_pred=knn.predict(model, Xtest)
    test_error=np.mean(y_pred.flatten() != ytest)
    print("The current test error is: %r" % test_error)

# part 3: plot classification boundaries for k=1 (use utils.plot_2dclassifier)
model1=knn.fit(X, y, k[2])
utils.plot_2dclassifier(model1, X, y)
#plt.show()

# save figure
fname = "../s16-c20-mean.png"
plt.savefig(fname)












