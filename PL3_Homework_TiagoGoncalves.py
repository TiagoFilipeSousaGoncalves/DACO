# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:29:25 2017

@author: Tiago Filipe Sousa Gonçalves DACO Student nºup201607753 FEUP
"""

#Homework PL3 Tiago Gonçalves
#Imported libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import datasets
from collections import Counter
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


#Dataset import
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(return_X_y=True)
features = data[0]
targets = data[1]
print(features.shape, targets.shape)

#Class imbalance
print('Positive Samples Proportion', np.sum(targets==1)/np.size(targets))

#Features normalization
features = StandardScaler().fit_transform(features)

#Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=10)


###############################################################################
#Naive Bayes

clf = GaussianNB()
clf.fit(X_train, y_train)
test = clf.predict(X_test)

accuracy_naivebayes = accuracy_score(test, y_test)
print('Accuracy of Naive Bayes is: \n', accuracy_naivebayes)


###############################################################################
#Logistic Regression
regularization_params = [0.0001, 0.001, 0.01, 1, 10]
for C in regularization_params:
    print('Evaluating parameter', C)
    clf_LR = LogisticRegression(C=C)
    cv_LR_scores = cross_val_score(clf_LR, X_train, y_train, cv=10)
    print('Mean accuracy across folds:', np.mean(cv_LR_scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (cv_LR_scores.mean(), cv_LR_scores.std() * 2))

#The best evaluation parameter was 1, so we can do final performance measurement
optimal_regularization_LR = 1
clf_LR = LogisticRegression(C=optimal_regularization_LR)
clf_LR.fit(X_train, y_train)
faccuracy_LR = clf_LR.score(X_test, y_test)
print('Accuray for the final performance measurement for LR is: \n', faccuracy_LR)


