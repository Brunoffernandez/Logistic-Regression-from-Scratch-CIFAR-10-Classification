#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 12:11:28 2021

@author: Quentin Billaudaz 6613136, Bruno Fernandez 6611648, Olle Settergren 6612873
"""

#import packages

import numpy as np
import matplotlib.pyplot as plt

 
# n denotes the sample size 
n = 100


# we simulate two types of features
half_sample=int(n/2)
x1 = np.random.multivariate_normal([-0.5, 1], [[1, 0.7],[0.7, 1]], half_sample)
x2 = np.random.multivariate_normal([2, -1], [[1, 0.7],[0.7, 1]], half_sample)
simulated_features = np.vstack((x1, x2)).astype(np.float64)

# the underlying value of beta in the simulation; the value we want to retrieve in the estimation procedure
beta_star=np.array([0.2,-0.8])



#The logistic function
def logistic(x):
    return 1 / (1 + np.exp(-x))


# Simulate the labels
def logistic_simulation(features,beta):    
    signal = np.dot(features, beta)
    p=logistic(signal)
    y= np.array([np.random.binomial(1, p[i] ) for i in range(n)])
    return y
 
#Skeleton for function Newton-Raphson for logisic regression

def logistic_regression_NR(features, target, num_steps, tolerance):
    #initialization of beta
    beta = np.zeros(features.shape[1])
    # Features is x matrix
    # Target is y matrix
    
    
    for step in range(num_steps):  
        signal = np.dot(features, beta)
        p = logistic(signal)   
        gradient = features.T @ (target - p)
        if np.linalg.norm(gradient) < tolerance:
            return beta, step
            
        #compute gradient and hessian
        
        W = np.diag(p * (1 - p))
        beta += np.linalg.inv(features.T @ W @ features) @ features.T @ (target - p)
                
    return beta, -1


simulated_labels = logistic_simulation(simulated_features, beta_star)

beta, step = logistic_regression_NR(simulated_features, simulated_labels, 100, 1e-12)
print(f'Estimated beta: {beta}')

#### Scatter plot of the features and correspoding labels
plt.figure(figsize=(12,8))
plt.scatter(simulated_features[:, 0], simulated_features[:, 1], c = simulated_labels, alpha = .5)
plt.show()


## Simulation study
S=1000
MLE_results = np.zeros((S, 2))
for i in range(S):

    #generate labels y for every simulation
    simulated_labels = logistic_simulation(simulated_features, beta_star)
    #compute the MLE for every simulation
    MLE, step = logistic_regression_NR(simulated_features, simulated_labels, 100, 1e-12)
    MLE_results[i] = MLE

beta_1 = np.mean(MLE_results[:,0])
beta_2 = np.mean(MLE_results[:,1])
print(f'Beta 1: {beta_1}')
print(f'Beta 2: {beta_2}')
#compute the means of estimated parameters beta_1 and beta_2
#make a histogram for the MLE of beta_1 and beta_2

plt.figure()
plt.hist(MLE_results[:,0],bins=20, label='beta_1')
plt.axvline(beta_1, color=(0.0, 0.0, 0.5, 1.0), label=(f'Mean: {beta_1}'))
plt.hist(MLE_results[:,1],bins=20, label='beta_2')
plt.axvline(beta_2, color=(0.5, 0.0, 0.0, 1.0), label=(f'Mean: {beta_2}'))
plt.title(f'Simulation study (S = {S}, n = {n})')
plt.legend()
plt.show()