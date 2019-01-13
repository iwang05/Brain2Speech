# use neural network
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import EEG_load
import EEG_feature_extraction

# subject_name = ["s05","s06","s07","s08","s09","s10","s11","s12","s13","s14","s15","s16","s17","s18","s19","s20","s21"]
result = []

for channel_number in range(1, 67):
    dataset = EEG_load.load_data("s21", channel_number)
    #dataset = EEG_feature_extraction.generate_feature_data("s10",channel_number)

    X = dataset['X_train']
    y = dataset['Y_train']
    Xtest = dataset['X_test']
    ytest = dataset['Y_test']

    # A binary classfier (use softmax for multi-classification)
    # 1.define a model (network)
    model = Sequential()
    # model.add(Dense(12, input_dim=1537, activation='relu'))
    model.add(Dense(40, input_dim=1537, activation='relu'))
    # model.add(Dense(40, input_dim=360, activation='relu'))
    # model.add(Dense(12, input_dim=2, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.add(Dropout(0.5))

    # 2.compile network
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    # 3.fit network (train)
    # history = model.fit(X, y, nb_epoch=100, batch_size=20)

    # how about using whole dataset instead of batches
    history = model.fit(X, y, nb_epoch=100, batch_size=None)

    # 4.evaluate network
    loss, accuracy = model.evaluate(Xtest, ytest)
    print("Channel" + str(channel_number))
    print("\nLoss: %.2f, Accuracy: %.2f%%" % (loss, accuracy * 100))

    result.append(float(round(accuracy * 100, 2)))

for i in result:
    print(i)
