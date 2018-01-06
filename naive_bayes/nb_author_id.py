#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB() # create a classifer
# to see how long it takes to train the classifer
t0 = time()
clf.fit(features_train, labels_train) # train the classifer with training data (features, and known labels)
print "training time: ", round(time()-t0, 3), "s"

# to see how long it takes to make prediction
t1 = time()
pred = clf.predict(features_test) # use trained classifer to predict the label of new data
print "predicted: ", pred
print "predicting time: ", round(time()-t1, 3), "s"

# Accuracy = # of test points that were correctly classfied / total number of test data
# Two ways of calculating accuracy:
# Method 1:
accuracy1 = clf.score(features_test, labels_test)
print "accuracy1:", accuracy1
# Method 2:
from sklearn.metrics import accuracy_score
accuracy2 =accuracy_score(labels_test, pred)
print "accuracy2: ", accuracy2


#########################################################


