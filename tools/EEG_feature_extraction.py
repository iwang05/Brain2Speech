#feature extraction for EEG signal
import math
import numpy as np
import pandas as pd
from scipy.signal import butter, lfilter, freqz, periodogram
from scipy.stats import kurtosis, skew
import scipy.io
import spectrum
import peakutils
import matplotlib.pyplot as plt
from sklearn import preprocessing
import random

########## band-pass filter ##########
def butter_bandpass(lowcut,highcut,fs,order=5):
	nyq=0.5*fs
	low=lowcut/nyq
	high=highcut/nyq
	b,a=butter(order,[low,high],btype='band',analog=False)
	return b, a

def butter_bandpass_filter(data,lowcut,highcut,fs,order=5):
	b, a = butter_bandpass(lowcut,highcut,fs,order=order)
	y=lfilter(b,a,data)
	return y

##########low-pass filter##########
def butter_lowpass(cutoff,fs,order=5):
	nyq=0.5*fs
	normal_cutoff=cutoff/nyq
	b,a=butter(order,normal_cutoff,btype='low',analog=False)
	return b, a

def butter_lowpass_filter(data,cutoff,fs,order=5):
	b, a = butter_lowpass(cutoff,fs,order=order)
	y=lfilter(b,a,data)
	return y

##########high-pass filter##########
def butter_highpass(cutoff,fs,order=5):
	nyq=0.5*fs
	normal_cutoff=cutoff/nyq
	b,a=butter(order,normal_cutoff,btype='high',analog=False)
	return b, a

def butter_highpass_filter(data,cutoff,fs,order=5):
	b, a = butter_highpass(cutoff,fs,order=order)
	y=lfilter(b,a,data)
	return y

########## band-stop filter ##########
def butter_bandstop(lowcut,highcut,fs,order=5):
	nyq=0.5*fs
	low=lowcut/nyq
	high=highcut/nyq
	b,a=butter(order,[low,high],btype='bandstop',analog=False)
	return b, a

def butter_bandstop_filter(data,lowcut,highcut,fs,order=5):
	b, a = butter_bandstop(lowcut,highcut,fs,order=order)
	y=lfilter(b,a,data)
	return y


def EEG_filter_band(signal, fs=1024):
	'''
	INPUT: EEG signal (Recommend: after removing mean value and other noise filtering method)

	The default value of sampling frequency is 1024Hz (For the letter/pseudo-letter dataset.)

	RETURN: A dictionary including 6 frequency band EEG signal

	'''

	theta = butter_bandpass_filter(signal, 4, 7, fs)
	alpha = butter_bandpass_filter(signal, 8, 13, fs)
	beta_1 = butter_bandpass_filter(signal, 14, 24, fs)
	beta_2 = butter_bandpass_filter(signal, 25, 35, fs)
	gamma_1 = butter_bandpass_filter(signal, 36, 58, fs)
	gamma_2 = butter_bandpass_filter(signal, 62, 100, fs)

	return{'theta':theta, 'alpha':alpha, 'beta1':beta_1, 'beta2':beta_2, 'gamma1':gamma_1, 'gamma2':gamma_2}

# a 30Hz lowpass filter
def EEG_lowpass(signal, fs=1024):
	filtered_EEG = butter_lowpass_filter(signal, 30, fs)
	return filtered_EEG

# filter out 54-66 Hz noise
def EEG_bandstop(signal, fs=1024):
	filtered_EEG = butter_bandstop_filter(signal, 54, 66, fs)
	return filtered_EEG

# normalize the signal to zero mean and unit variance
def standardization(signal):
	return preprocessing.scale(signal)


# normalize to scale [a,b]
def normalization(data,a,b):
	#data is a ndarray
	max_diff = max(data) - min(data)
	diff = data - min(data)
	return a+((b-a)*diff/max_diff)


def zero_mean(signal):
	# signal is EEG signal (a vector)
	return signal - np.mean(signal)


def EEG_mean(signal):
	return np.mean(signal)


def EEG_std(signal):
	return np.std(signal)


def EEG_kurtosis(signal):
	return scipy.stats.kurtosis(signal,bias=False)


def EEG_skewness(signal):
	return scipy.stats.skew(signal,bias=False)


# return the amplitude envelope of hilbert analytical signal
def hilbert(signal):
	return abs(scipy.signal.hilbert(signal))


def Feature_Extraction(signal, fs = 1024):
#This function reads the data (x) and the sampling frequency fs, and extracts the feature-vectors(y)
#x should be a vector (1 * N), and y is a row-vctor whose elements are the features

    #Mean
    #sig_mean = np.mean(signal)

    #STD - standard deviation
    #sig_std = np.std(signal)

    # If we normalize the signal to zero mean and unit variance,
    # then we do not need to compute these two features

    # Signal Power (after standardization ! otherwise the value will be too big)
    # sig_power = np.mean(np.square(signal))

    #Kurtosis
    Kseg = scipy.stats.kurtosis(signal,bias=False)

	#Skewness
    Sseg = scipy.stats.skew(signal,bias=False)

    Feature = np.array([Kseg, Sseg])
    #Feature = np.array([sig_mean, sig_std])

    return Feature
	#return a numpy array row-vector


def generate_feature_data(subject_name, channel_number):
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

	X_feature = list()

	X_cut = X[:, 512:872]

	#normalization and band filtering
	for i in range(X.shape[0]):
		#X[i] = EEG_filter_band(X[i])['theta']
		#X[i] = EEG_filter_band(X[i])['alpha']
		#X[i] = EEG_filter_band(X[i])['beta1']
		#X[i] = EEG_filter_band(X[i])['beta2']
		#X[i] = EEG_filter_band(X[i])['gamma1']
		#X[i] = EEG_filter_band(X[i])['gamma2']
		#X_cut[i] = standardization(X_cut[i])
		#X_cut[i] = normalization(X_cut[i], -1, 1)
		X_feature.append(Feature_Extraction(X_cut[i]))

	X = np.array(X_feature)
	# convert data to feature vectors (N * 2 numpy array)

	# disorganize the data to split to training and testing set
	index = [i for i in range(len(X))]
	random.shuffle(index)
	X = X[index]
	Y = Y[index]

	num_train = round(0.8 * X.shape[0])

	X_train = X[0 : (num_train-1)]
	Y_train = Y[0 : (num_train-1)]
	X_test = X[num_train : (X.shape[0]-1)]
	Y_test = Y[num_train : (X.shape[0]-1)]

	return {"X_train":X_train, "Y_train": Y_train,"X_test": X_test, "Y_test": Y_test}



