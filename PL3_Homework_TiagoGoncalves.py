# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 11:29:25 2017

@author: Tiago Filipe Sousa Gonçalves DACO Student nºup201607753 FEUP
"""

#Homework PL3 Tiago Gonçalves
#Imported libraries
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn import tree
from sklearn.svm import SVC
from sklearn import metrics
import matplotlib.pyplot as plt

#Dataset import
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer(return_X_y=True)
features = data[0]
targets = data[1]
print(features.shape, targets.shape)

#Class imbalance
print('Positive Samples Proportion', np.sum(targets==1)/np.size(targets))
print()

#Features normalization
features = StandardScaler().fit_transform(features)

#Split into train and test set
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.20, random_state=10)


###############################################################################
#Naive Bayes
print('Naive Bayes')
clf = GaussianNB()
clf.fit(X_train, y_train)
test = clf.predict(X_test)


accuracy_naivebayes = accuracy_score(test, y_test)
print('Accuracy of Naive Bayes is: ', accuracy_naivebayes)

#With Cross Validation
print ('Using Cross Validation')
cv_NB_scores = cross_val_score(clf, X_train, y_train, cv=10)
print ('Mean accuracy across folds: ', np.mean(cv_NB_scores))
print()
###############################################################################
#Logistic Regression
print('Logistic Regression')
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
print('Accuracy for the final performance measurement for LR is: ', faccuracy_LR)
print()
###############################################################################
#kNN Neighbours
print('k-Nearest Neighbours')
n_neighbors = [1,2,3,4,5,6,7,8,9,10,11]

for n in n_neighbors:
    print('n-neighbours:', n)
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    knn_scores = cross_val_score(knn, X_train, y_train, cv=10)
    print('Mean accuracy across folds:', np.mean(knn_scores))
    print("Accuracy: %0.2f (+/- %0.2f)" % (knn_scores.mean(), knn_scores.std() * 2))

#The best n is n=4, so we can use to do a final performance measurement
optimal_number_neighbors = 4
knn = KNeighborsClassifier(n_neighbors = optimal_number_neighbors)
knn.fit(X_train, y_train)
kaccuracy = knn.score(X_test, y_test)
print('Accuracy for the final performance measurement for kNN is: ', kaccuracy)
print()
###############################################################################
#Support Vector Machines
print('Support Vector Machines')
#First we need to find the best parameters using GridSearchCV
print('Finding the best parameters...')
# Set the parameters by cross-validation
parameters = [{'kernel': ['rbf'],
               'gamma': [0.01, 0.1],
                'C': [1, 10, 100]},
              {'kernel': ['linear'], 'C': [1, 10, 100]}]

clf_svm = GridSearchCV(svm.SVC(decision_function_shape='ovr'), parameters, cv=6)
print("Tuning hyper-parameters")
clf_svm.fit(X_train[:10,:], y_train[:10])
print("Best parameters set found on validation set:")
print()
print(clf_svm.best_params_)
print()
print("Grid scores on training set:")
print()
means = clf_svm.cv_results_['mean_test_score']
stds = clf_svm.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clf_svm.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print('SVM Accuracy is:', clf_svm.score(X_test, y_test))
print()
#With Cross Validation
print ('Using Cross Validation')
cv_SVM_scores = cross_val_score(clf_svm, X_train, y_train, cv=10)
print ('Mean accuracy across folds: ', np.mean(cv_SVM_scores))
print()
###############################################################################
#Decision Trees
print('Decision Trees')
#Finding the best parameters
param_grid = {'criterion':['gini'],  
              'max_depth':[None, 2, 5, 10], 
              'min_samples_split':[2, 10, 20], 
              'min_samples_leaf':[1, 5, 10]}

clfd_tree = GridSearchCV(tree.DecisionTreeClassifier(), param_grid, cv=6)
print("Tuning hyper-parameters")
clfd_tree.fit(X_train[:10,:], y_train[:10])
print("Best parameters set found on validation set:")
print()
print(clfd_tree.best_params_)
print()
print("Grid scores on training set:")
print()
means = clfd_tree.cv_results_['mean_test_score']
stds = clfd_tree.cv_results_['std_test_score']
for mean, std, params in zip(means, stds, clfd_tree.cv_results_['params']):
    print("%0.3f (+/-%0.03f) for %r"
          % (mean, std * 2, params))
print('Decision Tree Accuracy is:', clfd_tree.score(X_test, y_test))
print()
X = X_train
Y = y_train
clf_dtree = tree.DecisionTreeClassifier()
clf_dtree = clf_dtree.fit(X, Y)
d_tree_pr = clf_dtree.predict(X_test)
dt_accuracy = clf_dtree.score(X_test, y_test)
print('Accuracy for Decision Trees is:', dt_accuracy)

#With Cross Validation
print ('Using Cross Validation')
cv_DT_scores = cross_val_score(clf_dtree, X_train, y_train, cv=10)
print ('Mean accuracy across folds: ', np.mean(cv_DT_scores))
print()

#Decision Trees with Regression
#I didn't perform hyperparameter tuning on this one
print('Decision Trees with Regression')
clf_dtr = tree.DecisionTreeRegressor()
clf_dtr = clf_dtr.fit(X, Y)
dtr_pr = clf_dtr.predict(X_test)
dtr_accuracy = clf_dtr.score(X_test, y_test)
print('Accuracy for Decision Trees Regression is:', dtr_accuracy)

#With Cross Validation
print ('Using Cross Validation')
cv_DTR_scores = cross_val_score(clf_dtr, X_train, y_train, cv=10)
print ('Mean accuracy across folds: ', np.mean(cv_DTR_scores))
print()

###############################################################################
#Getting more metrics on the best classifier, which was SVM
#Doing another performance with SVM
print('SVM The Best Classifier')
SVM_clf = SVC(C=1.0, gamma=0.1, probability=True, kernel='rbf')
SVM_clf = SVM_clf.fit(X_train, y_train) 
svm_pred = SVM_clf.predict(X_test)
#Compute accuracy for this SVM
svm_accuracy = SVM_clf.score(X_test, y_test)
print ('Accuracy for this SVM is:', svm_accuracy)
#Confusion Matrix
svm_cm = metrics.confusion_matrix(y_test, svm_pred)
print('Confusion Matrix is:')
print(svm_cm)
tn, fp, fn, tp = svm_cm.ravel()
print('True Negatives:', tn)
print('False Positives:', fp)
print('False Negatives:', fn)
print('True Positives:', tp)
#Sensitivity
svm_sens = metrics.recall_score(y_test, svm_pred)
print('Sensitivity is:', svm_sens)
#Specificity
svm_spec = tn / float(tn+fp)
print('Specificity is:', svm_spec)
#Precision
svm_prc = metrics.precision_score(y_test, svm_pred)
print('Precision is:', svm_prc)
#F1 - Score
svm_f1 = metrics.f1_score(y_test, svm_pred)
print('F1 Score is:', svm_f1)
#Complete report
print('Complete Report:')
print(metrics.classification_report(y_test, svm_pred, target_names=['Malignant', 'Benign']))

#Decision Threshold and ROC Curve
#Store predicted probabilities for class 1 - Malignant
svm_pred_proba = SVM_clf.predict_proba(X_test)[:,0]
#Probability Histogram
plt.hist(svm_pred_proba, bins=8)
# x-axis limit from 0 to 1
plt.xlim(0,1)
plt.title('Histogram of predicted probabilities');
plt.xlabel('Predicted probability of breast cancer');
plt.ylabel('Frequency');
fpr, tpr, thresholds = metrics.roc_curve(y_test, svm_pred_proba)

#Plot ROC Curver
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC curve for breast cancer classifier')
plt.xlabel('False Positive Rate (1 - Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

#Area Under ROC Curve
print('Area Under ROC Curve is:')
print(metrics.roc_auc_score(y_test, svm_pred_proba))


#Calculate precision at the default threshold decision
# define a function that accepts a threshold and prints sensitivity and specificity
def evaluate_threshold(threshold):
    print('Sensitivity:', tpr[thresholds > threshold][-1])
    print('Specificity:', 1 - fpr[thresholds > threshold][-1])

print('threshold =', 0.5)    
evaluate_threshold(0.5)