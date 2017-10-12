# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:21:38 2017

@author: tiago
"""

#Homework PL2 Tiago Gonçalves
#6. Homework
#6.1 - Graded Homework: Training and Testing data.

#Import libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

#Create function (Provided by Professor)
def print_predictions_and_labels(array_preds_labels):
    for predicted_pair in array_preds_labels:
        prediction = predicted_pair[0]
        label = predicted_pair[1]
        print('Prediction', prediction, 'Label', label)


#Part 1
#Read Data
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data',
                 header=None)
df.columns = ['#pregnant', 'glucose', 'blood_pressure', 'thickness', 'insulin', 'body_mass_idx', 
              'pedigree', 'age', 'label']
df.head()

#Separate numpy arrays for labels and features
dataset = np.array(df)
print(df.shape)

features = dataset[:, :8]
labels = dataset[:, -1]

#print(labels)

#Reduce features to easily implement Naive Bayes
#df_reduced = df[['body_mass_idx','age']]
#features_reduced = np.array(df_reduced)

#Split data into train set and test set
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.25)

#Fit model with sklearn (Training classifier)
#Use test set
clf = GaussianNB()
clf.fit(X_train, y_train)
test = clf.predict(X_test)

#Test classifier (with test set)
#print(test)

#Store test predictions and labels in a numpy array
new_array = np.concatenate([test.reshape(-1, 1), y_test.reshape(-1, 1)], axis=1)
print_predictions_and_labels(new_array)

#Part 2
#Alternative 1: 
accuracy = accuracy_score(test, y_test, normalize=False)/len(y_test)
print(accuracy)
#Alternative 2: 
accuracy_sklearn = accuracy_score(test, y_test)
print(accuracy_sklearn)
# Alternative 3: 
accuracy_sklearn_2 = (clf.fit(X_train, y_train)).score(X_test, y_test)
print(accuracy_sklearn_2)