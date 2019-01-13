# use CNN(ConvNet)
from keras.models import Sequential
from keras.layers import Dense,Activation,LSTM,Dropout
import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
import EEG_load
import EEG_feature_extraction

#subject_number = ["s09","s10","s11","s12","s13","s14","s15","s17","s18","s19","s20","s21"]
result=[]

for channel_number in range(1,67):

	#dataset = EEG_feature_extraction.generate_feature_data("s10",channel_number)
	dataset = EEG_load.load_data("s21",channel_number)
	
	X = dataset['X_train']
	y = dataset['Y_train']
	Xtest = dataset['X_test']
	ytest = dataset['Y_test']

	X = np.reshape(X, (X.shape[0], 1, X.shape[1]))
	Xtest = np.reshape(Xtest, (Xtest.shape[0], 1, Xtest.shape[1]))

	# A binary classfier (use softmax for multi-classification)
	# 1.define a model (network)
	model = Sequential()

	model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))
	#model.add(Dropout(0.5))
	model.add(Dense(1, activation='sigmoid'))
	#model.add(Dropout(0.5))

	# 2.compile network
	model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

	# 3.fit network (train)
	history = model.fit(X, y, nb_epoch = 100, batch_size = None)

	# 4.evaluate network
	loss, accuracy = model.evaluate(Xtest,ytest)
	print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

	result.append(float(round(accuracy*100,2)))

for i in result:
	print(i)


# create and fit the model
#model = Sequential()
#model.add(LSTM(32, input_shape=(X.shape[1], X.shape[2])))

# Multi-class classification
#model.add(Dense(y.shape[1], activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#model.fit(X, y, nb_epoch=500, batch_size=1, verbose=2)


