# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 07:44:54 2019

@author: Zain
"""

from sklearn.svm import SVC
from sklearn.model_selection import KFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib as plt
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import KFold
from sklearn.tree import export_graphviz

scaler = MinMaxScaler()
dataset = pd.read_csv('CSV_DATA/GLCM_8Z_BrainTumor_Data.csv')
#dataset = dataset.sample(frac=1).reset_index(drop=True)
X = dataset.iloc[:, 2:].values
Y = dataset.iloc[:, 1].values

x_train,x_test,y_train,y_test=train_test_split(X,
                                               Y,
                                               train_size=0.8,
                                               test_size=0.2)
def Average(lst): 
    return sum(lst) / len(lst) 

List = ["contrast","energy","homogeneity","correlation","dissimilarity","ACM"]
List1 = ["Zone1","Zone2","Zone3","Zone4","Zone5","Zone6","Zone7","Zone8"]
features_name = []
class_names = ["B","M","N"]
for i in List1:
    for j in List:
        features_name.append(str(i)+" "+str(j))
# 14 5 sgd 81 %
"""                                           
clf = MLPClassifier(alpha=1e-9,
                    max_iter=10000,
                    hidden_layer_sizes=(1000,1000,1000),
                    solver='sgd',
                    shuffle=False,
                    tol=1e-9)

clf.fit(x_train,y_train) 
pred = clf.predict(x_test) 
"""
from sklearn.metrics import average_precision_score
pred = []
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=10)
Avgacc = []
clf.fit(x_train,y_train)
estimator = clf.estimators_[5]
kf = KFold(n_splits=10)
kf.get_n_splits(X)

precisionlist = []
recalllist = []
fscorelist = []

F_CF = [[0,0,0],[0,0,0],[0,0,0]]
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, pred, normalize=True)
    Avgacc.append(accuracy)
    precision,recall,fscore,support = precision_recall_fscore_support(y_test, pred, average='macro')
    precisionlist.append(precision)
    recalllist.append(recall)
    fscorelist.append(fscore)
    
    conf_mat = confusion_matrix(pred, y_test)
    F_CF += conf_mat
    #print(precision)
    #average_precision = average_precision_score(y_test, pred)

    #print('Average precision-recall score: {0:0.2f}'.format(average_precision))
    
    #print("\n\n\n\nAccuracy: "+str(round(accuracy*100)) +" %")

#str_tree = export_graphviz(estimator, out_file='tree.dot', feature_names = features_name,class_names = class_names,rounded = True, proportion = False, precision = 2, filled = True)

#from IPython import display
#display.display(str_tree)
print("\n\n\nPrecision: {} \nRecall: {} \nf-Score: {}".format(sum(precisionlist)/4,sum(recalllist)/4,sum(fscorelist)/4))
print("Accuracy: "+str(round(Average(Avgacc)*100)) +" %")    
"""
from sklearn import tree
clf = tree.DecisionTreeRegressor()
clf = clf.fit(x_train,y_train)
pred = clf.predict(x_test)
"""

"""
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(x_train,y_train) 
pred = clf.predict(x_test)
"""


"""
accuracy = accuracy_score(y_test, pred, normalize=True)
precision,recall,fscore,support = precision_recall_fscore_support(y_test, pred, average='macro')

print("\n\n\n\nAccuracy: "+str(round(accuracy*100)) +" %")
print("Precision: "+str(precision))
print("Recall: "+str(recall))
print("FScore: "+str(fscore)+"\n\n\n\n\n")
"""

color = ["skyblue","lightskyblue","deepskyblue","deepskyblue","mediumslateblue","blue",]
import matplotlib.pyplot as plt
names = ['SVM', 'Naive Bayes' ,'Neural Network', 'Decision Tree', "Random Forest", "Random Forest K Fold"]
values = [58, 60, 72,73,82,96]
plt.xlabel("\n\nUsed Algorithms", fontsize=25)
plt.ylabel("\n\nAccuracy in Percentage", fontsize=25)
plt.rcParams.update({'font.size': 20})
plt.bar(names, values, width=0.6,color = color)


actuall = []
predictedd = []
for i in y_test:
    actuall.append(int(i))
for i in pred:
    predictedd.append(int(i))


import ml_metrics 
import numpy as np
ml_metrics.mapk(actuall, predictedd)


conf_mat = confusion_matrix(pred, y_test)
arr = sum(F_CF)

print("\n\n\n\nLabel Wise Accuracy")
print("Label 0: {} % \nLabel 1: {} % \nLabel 2: {} %".format(603*100/666,1316*100/1371,834*100/900))









color = ["pink","red",]
import matplotlib.pyplot as plt
names = ['Deep CNN-SVM', "Zone Based Classification"]
values = [97.1, 97.5]
plt.xlabel("\nSpecificity", fontsize=25)
plt.rcParams.update({'font.size': 18})
plt.bar(names, values, width=0.2,color = color)






















