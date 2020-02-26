# coding = utf-8
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import train_test_split
from sklearn.model_selection import learning_curve

from sklearn.datasets import load_digits

dataset = load_digits()
print 'dataset.keys = ',dataset.keys()

X = dataset['data']
y = dataset['target']
plt.imshow(X[8].reshape(8,8))
plt.show()
reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=1))
scores_ada = cross_val_score(reg_ada, X, y, cv=6)
print 'scores_ada.mean = ',scores_ada.mean()


score = []
for depth in [1,2,10,100] : 
    reg_ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=depth))
    scores_ada = cross_val_score(reg_ada, X, y, cv=6)
    score.append(scores_ada.mean())
print 'score = \n',score


