# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:32:14 2019

@author: udprajapati
"""
#Imort Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#Import Dataset
dataset = pd.read_csv(r'C:\uday\ML\Coursera_ML_ANDREW_NG\machine-learning-ex1\ex1\data.csv')

#Split fetaure and output variable
X = dataset.iloc[:, [0]].values
Y = dataset.iloc[:, [1]].values



def computeCost(X, Y, theta):
    m = len(Y)
    Y_pred = np.dot(X, theta)
    J = (1/(2*m)) * np.sum(np.square(Y_pred - Y))
    return J

def gradientDescent(X, Y, theta, alpha, iterations):
    m = len(Y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    for i in range(iterations):
        Y_pred = np.dot(X, theta)
        theta = theta - (1/m)*alpha*(np.dot(X.T, (Y_pred - Y)))
        theta_history[i, :] = theta.T
        cost_history[i] = computeCost(X, Y, theta)
    return theta, cost_history
        

#Const function
X = np.append(np.ones((len(X), 1), np.int8), X, axis = 1)
theta = np.zeros((2, 1))
cost_val = computeCost(X, Y, theta)
cost_val1 = computeCost(X, Y, [[-1], [2]]);

#Gradient descent
iterations = 1500;
alpha = 0.01;
theta, cost_history = gradientDescent(X, Y, theta, alpha, iterations);

Y_pred = np.dot(X, theta)

plt.scatter(X[:, 1], Y, color='red')
plt.xlabel('Population')
plt.ylabel('Population of City in 10,000s')
plt.plot(X[:, 1], Y_pred, color='blue')
plt.show()

predict1 = np.dot([[1, 3.5]], theta);
predict2 = np.dot([[1, 7]], theta);



