#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 18 15:11:32 2021

@author: ctheobal
"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras

'''
This codes trains a feed forward neural network on the iris dataset.
Required python libraires: sklearn, tensorflow, matplotlib
'''

iris = datasets.load_iris() # Loading the iris dataset

X = iris.data # Retrieving the data and the labels (classes)
Y = iris.target

print(X.shape)
print(Y.shape)

Y = Y.reshape(-1,1) # Reshape to use the one hot encoder: it needs to be two dimensional
print(Y.shape)
enc = OneHotEncoder() # One hot encoder to vectorize the classes
Y = enc.fit_transform(Y).toarray()
print("Mapped output classes:")
print(enc.categories_)
print("New label shape:")
print(Y.shape)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=1) # Split train and test set randomly with 75% and 25% proportion respectively
print(X_train.shape)
print(X_test.shape)

def create_model():
    '''
    Create a fully connected neural network with one input layer, one hidden layer and one output layer with softmax activation.
    '''
    input_layer = keras.layers.Input(shape=(4,)) # Input layer
    h = keras.layers.Dense(128)(input_layer) # Hidden layer
    h = keras.layers.Dense(3, activation='softmax')(h) # Output layer
    model = keras.models.Model(inputs=input_layer, outputs=h)
    return model

model = create_model() # We create the model
model.compile(optimizer=keras.optimizers.SGD(lr=1e-2), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model: define optimizer, loss and eventual metrics

# Fit the model: define batch size, number of epochs
model_history = model.fit(X_train,Y_train,
                          batch_size=16,
                          epochs=250,
                          validation_data=(X_test,Y_test))

print("Final train accuracy is: "+str(model_history.history["accuracy"][-1]))
print("Final test accuracy is: "+str(model_history.history["val_accuracy"][-1]))

# Plot results: accuracy and loss

t = range(250)

plt.plot(t,model_history.history["accuracy"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_accuracy"],'b', label='test', linewidth=1.5)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig('accuracy.pdf')
plt.clf()

plt.plot(t,model_history.history["loss"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_loss"],'b', label='test', linewidth=1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.savefig('loss.pdf')
plt.clf()


## Lower learning rate

learning_rate = 1e-4

model = create_model() # We create the model
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model: define optimizer, loss and eventual metrics

model_history = model.fit(X_train,Y_train,
                          batch_size=16,
                          epochs=250,
                          validation_data=(X_test,Y_test))

print("Final train accuracy is: "+str(model_history.history["accuracy"][-1]))
print("Final test accuracy is: "+str(model_history.history["val_accuracy"][-1]))

plt.plot(t,model_history.history["accuracy"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_accuracy"],'b', label='test', linewidth=1.5)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

plt.plot(t,model_history.history["loss"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_loss"],'b', label='test', linewidth=1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()


## Higher learning rate

learning_rate = 1

model = create_model() # We create the model
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model: define optimizer, loss and eventual metrics

model_history = model.fit(X_train,Y_train,
                          batch_size=16,
                          epochs=250,
                          validation_data=(X_test,Y_test))

print("Final train accuracy is: "+str(model_history.history["accuracy"][-1]))
print("Final test accuracy is: "+str(model_history.history["val_accuracy"][-1]))

plt.plot(t,model_history.history["accuracy"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_accuracy"],'b', label='test', linewidth=1.5)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

plt.plot(t,model_history.history["loss"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_loss"],'b', label='test', linewidth=1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

## No hidden layer

def create_model():
    '''
    Create a fully connected neural network with one input layer and one output layer with softmax activation.
    '''
    input_layer = keras.layers.Input(shape=(4,)) # Input layer
    h = keras.layers.Dense(3, activation='softmax')(input_layer) # Output layer
    model = keras.models.Model(inputs=input_layer, outputs=h)
    return model

learning_rate = 1e-2

model = create_model() # We create the model
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model: define optimizer, loss and eventual metrics

model_history = model.fit(X_train,Y_train,
                          batch_size=16,
                          epochs=250,
                          validation_data=(X_test,Y_test))

print("Final train accuracy is: "+str(model_history.history["accuracy"][-1]))
print("Final test accuracy is: "+str(model_history.history["val_accuracy"][-1]))

plt.plot(t,model_history.history["accuracy"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_accuracy"],'b', label='test', linewidth=1.5)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

plt.plot(t,model_history.history["loss"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_loss"],'b', label='test', linewidth=1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

## Multiple layers

def create_model():
    '''
    Create a fully connected neural network with one input layer and one output layer with softmax activation.
    '''
    input_layer = keras.layers.Input(shape=(4,)) # Input layer
    h = keras.layers.Dense(128)(input_layer) # Hidden layer
    h = keras.layers.Dense(128)(h) # Hidden layer
    h = keras.layers.Dense(128)(h) # Hidden layer
    h = keras.layers.Dense(3, activation='softmax')(h) # Output layer
    model = keras.models.Model(inputs=input_layer, outputs=h)
    return model

learning_rate = 1e-2

model = create_model() # We create the model
model.compile(optimizer=keras.optimizers.SGD(lr=learning_rate), loss='categorical_crossentropy', metrics=['accuracy']) # Compile the model: define optimizer, loss and eventual metrics

model_history = model.fit(X_train,Y_train,
                          batch_size=16,
                          epochs=250,
                          validation_data=(X_test,Y_test))

print("Final train accuracy is: "+str(model_history.history["accuracy"][-1]))
print("Final test accuracy is: "+str(model_history.history["val_accuracy"][-1]))

plt.plot(t,model_history.history["accuracy"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_accuracy"],'b', label='test', linewidth=1.5)
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()

plt.plot(t,model_history.history["loss"],'r', label='train', linewidth=1.5)
plt.plot(t,model_history.history["val_loss"],'b', label='test', linewidth=1.5)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend()
plt.show()
plt.clf()