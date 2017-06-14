#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi","salary","from_this_person_to_poi"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)

from sklearn import cross_validation

features_train,features_test,labels_train,labels_test = cross_validation.train_test_split(features,labels,test_size=0.3,random_state=42)


### it's all yours from here forward!  
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import numpy 

clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train,labels_train)

## How many POI's in the test set
list_1 = []
for p in labels_test:
    if p == 1.0:
        list_1.append(p)
    
print len(list_1)
## How many total people in the test set?
print len(labels_test)

## Compare test set (actual set) with predictions
print labels_test
pred =  clf.predict(features_test) 
print pred


## What’s the precision?
precision_score = precision_score(labels_test,pred)
print precision_score

## Whats the recall score?
recall_score = recall_score(labels_test,pred)
print recall_score

acc = clf.score(features_test,labels_test)
print acc



