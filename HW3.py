import torch
import numpy as np
from scipy import spatial

from keras.models import Sequential
from keras.layers import Dense, Input, Dropout, SimpleRNN,LSTM, Activation, Embedding
from keras.utils import np_utils

import matplotlib.pyplot as plt

import emoji

import data_train_test.emo_utils as em
import fucntions as fn


if __name__ == "__main__":
    # X,Y = em.read_csv(filename='data_train_test/emojify_data.csv')
    w2i, i2w, w2v = em.read_glove_vecs(glove_file='glove.6B/glove.6B.50d.txt') #출력이 Dict 형식임
    
    X_train, Y_train = em.read_csv(filename='data_train_test/train_emoji.csv')
    X_test, Y_test = em.read_csv(filename='data_train_test/test_emoji.csv')

    Y_train = fn.onehot(Y_train)
    Y_test = fn.onehot(Y_test)

    # print (X_train.shape, Y_train.shape, X_test.shape, Y_test.shape)

    # Splitting the sentences
    X_train_vec = []
    for i in range(X_train.shape[0]):
        X_train_vec.append(X_train[i].split())
    X_train_vec = np.array(X_train_vec)   

    X_test_vec = []
    for i in range(X_test.shape[0]):
        X_test_vec.append(X_test[i].split())
    X_test_vec = np.array(X_test_vec)

    # # Check Dataset Dimmension
    # print (X_train_vec[0],Y_train[0])

    # Check maximum length of dataset sentence.
    np.unique(np.array([len(ix) for ix in X_train_vec]) , return_counts=True)
    np.unique(np.array([len(ix) for ix in X_test_vec]) , return_counts=True)

    # # Checking Spatial distance happy to sad.
    # print(spatial.distance.cosine(w2v["happy"], w2v["sad"]))

    # Creating Embedding Matrix
    embedding_matrix_train = np.zeros((X_train_vec.shape[0], 10, 50))
    embedding_matrix_test = np.zeros((X_test_vec.shape[0], 10, 50))

    for ix in range(X_train_vec.shape[0]):
        for ij in range(len(X_train_vec[ix])):
            embedding_matrix_train[ix][ij] = w2v[X_train_vec[ix][ij].lower()]
        
    for ix in range(X_test_vec.shape[0]):
        for ij in range(len(X_test_vec[ix])):
            embedding_matrix_test[ix][ij] = w2v[X_test_vec[ix][ij].lower()]

    # Check Shape of Embedding Matrix
    print (embedding_matrix_train.shape, embedding_matrix_test.shape)

#--------------- Data preprocessing ---------------

    # # A simple RNN network 

    # model = Sequential()
    # model.add(SimpleRNN(64, input_shape=(10,50), return_sequences=True))
    # model.add(Dropout(0.5))
    # model.add(SimpleRNN(64, return_sequences=False))
    # model.add(Dropout(0.5))
    # model.add(Dense(5))
    # model.add(Activation('softmax'))

    # # Check model structure
    # model.summary()

    # A LSTM network

    model = Sequential()
    model.add(LSTM(64,input_shape=(10,50), return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(5))
    model.add(Activation('softmax'))

    # Check model structure
    model.summary()

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    hist = model.fit(embedding_matrix_train,Y_train,
                epochs = 50, batch_size=32,shuffle=True
                )

    # Prediction trained model 
    pred = model.predict_classes(embedding_matrix_test)

    # print(pred)
    # print(np.argmax(Y_test,axis=1))
#--------------- Model Structure ---------------
    # # Calculating the accuracy of the algorithm
    # print(float(sum(pred==Y_test))/embedding_matrix_test.shape[0])

    print(fn.accuracy(np.argmax(Y_test,axis=1),pred,embedding_matrix_test))

    
    # # Printing the sentences with the predicted and labled emoji
    # for ix in range(embedding_matrix_test.shape[0]):
        
    #     if pred[ix] != Y_test[ix]:
    #         print(ix)
    #         print (test[0][ix],end=" ")
    #         print (emoji.emojize(emoji_dict[pred[ix]], use_aliases=True),end=" ")
    #         print (emoji.emojize(emoji_dict[Y_test[ix]], use_aliases=True))

#--------------- Model Accuracy ---------------