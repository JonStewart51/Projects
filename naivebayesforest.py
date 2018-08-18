# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 11:39:20 2018

@author: JonStewart51
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA, KernelPCA
import math
#sklearn.decomposition.FastICA
from sklearn.decomposition import FastICA
from itertools import combinations_with_replacement
from sklearn.naive_bayes import GaussianNB
#gnb = GaussianNB()
import numpy as np
import pandas as pd
import math
import sys


class nbforest():
    """
    Naive Bayes Forest: Random Subspace aggregate models using naive bayes as base learner.
    Good for very noisy data, typically more bias than if trees are used as base learners
    -this has ICA build in to ensure feature independence. If the variables are truly
    gaussian distributed (can be described entirely by the first two
    statistical moments), then pca/kpca can and should be used instead.
    init:
        n_estimators: number of base models
        max_features: number of variables to include in each model
        
    """
    def __init__(self, n_estimators=10, max_features=50):
                 
        self.n_estimators = n_estimators    # Number of trees
        self.max_features = max_features    # Maxmimum number of features per tree
        self.indices = []
        self.FastICAlist = []
        for _ in range(n_estimators):
            self.FastICAlist.append(FastICA())
        self.gaussiannb_list = []
        for _ in range(n_estimators):
            self.gaussiannb_list.append(
                GaussianNB()) #set random state to ensure consistency of results

    def fit(self, X, Y):
        
        n_features = np.shape(X)[1]  #this is changed from a standard random forest. Here, we want to have similar trees, split over samples rather than features.
        
        if not self.max_features:
            self.max_features = int(math.sqrt(n_features))
        for i in range(self.n_estimators):
            #X_subset, y_subset = subsets[i]
            # Feature bagging (select random subsets of the features)
            idx = np.random.choice(range(n_features), size=self.max_features, replace=True)
            # Save the indices of the features for prediction
            self.indices.append(idx)
            # Choose the features corresponding to the indices
            X_subset = X[:,idx] #reversed from original
            #y_subset = y[idx]
            # Fit the tree to the data
            X_ica = self.FastICAlist[i].fit_transform(X_subset)
            self.gaussiannb_list[i].fit(X_ica, Y)
            
    def predict(self, X):
        X_predlist = []
        
        for i, nb in enumerate(self.gaussiannb_list):
            idx = self.indices[i]
            X_new = X[:,idx]
            X_ica = self.FastICAlist[i].transform(X_new)
            X_pred = self.gaussiannb_list[i].predict(X_ica)
            X_predlist.append(X_pred)
            
        X_predarray = np.asarray(X_predlist)
        #now, do row sum, divide by number of rows. If >.5, then 1, else 0
        X_avg = np.sum(X_predarray, axis = 0) / self.n_estimators
        X_predictions = np.where(X_avg < .5, 0, 1)
        return(X_predictions)
        
###Quick Example
X = pima_diabetescsv[:,0:8]
Y = pima_diabetescsv[:,8]  

test1 = nbforest(10)        
test2 = test1.fit(X, Y)       
test3 = test1.predict(X)    

comp = np.vstack((test3, Y)) 
comp1 = np.sum(comp, axis = 0)
comp2 = np.where(comp1 == 1, 0, 1)
comp3 = np.mean(comp2)





