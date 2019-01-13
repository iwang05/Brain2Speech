# load dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import random
import EEG_feature_extraction
from sklearn import preprocessing

# normalize the signal to zero mean and unit variance
def standardization(signal):
	return preprocessing.scale(signal)

def normalization(data,a,b):
	#data is a ndarray
	max_diff = max(data) - min(data)
	diff = data - min(data)
	return a+((b-a)*diff/max_diff)

def load_data(subject_name,channel_number):
	data_dir = "/Users/irene/PycharmProjects/Brain2Speech/data/" + subject_name + "/" + subject_name + "_channel"
	# + number_of_cannel + ".csv"

	# only use one channel data
	data = pd.read_csv(data_dir+str(channel_number)+".csv")

	# read data from all channel (1-66)
	#for j in range(2,67):
	#	new_data = pd.read_csv(data_dir + str(j) +".csv")
	#	data = pd.concat([data, new_data], axis=1)


	data_T = data.T 
	#the last column are the labels (0-letter; 1-pseudo-letter)

	nrow = data_T.shape[0]
	ncol = data_T.shape[1]

	#the first line (row) should be discarded (get [1:231] lines)
	data_trim = data_T[1:nrow]

	X = data_trim.drop(ncol-1,axis=1)
	Y = data_trim[[ncol-1]]

	X=np.array(X)
	Y=np.array(Y)

	# disorganize the data to split to training and testing set
	index = [i for i in range(len(X))]
	random.shuffle(index)
	X = X[index]

	#X_cut = X[:, 512:872]
	# normalize to [-1,1]scale
	for i in range(X.shape[0]):
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['theta']
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['alpha']
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['beta1']
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['beta2']
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['gamma1']
		#X[i] = EEG_feature_extraction.EEG_filter_band(X[i])['gamma2']
		#X[i] = EEG_feature_extraction.EEG_lowpass(X[i])

		#X_cut[i] = standardization(X_cut[i])
		#X_cut[i] = normalization(X_cut[i],-1,1)
		#X[i] = EEG_feature_extraction.EEG_bandstop(X[i])

		X[i] = standardization(X[i])
		X[i] = normalization(X[i], -1, 1)
	#X = normalization(X, -1, 1)
	Y = Y[index]

	num_train = round(0.8 * X.shape[0])

	X_train = X[0 : (num_train-1)]
	#X_train = X_cut[0 : (num_train-1)]
	Y_train = Y[0 : (num_train-1)]
	X_test = X[num_train : (X.shape[0]-1)]
	#X_test = X_cut[num_train : (X.shape[0]-1)]
	Y_test = Y[num_train : (X.shape[0]-1)]

	return {"X_train":X_train, "Y_train": Y_train,"X_test": X_test, "Y_test": Y_test}
