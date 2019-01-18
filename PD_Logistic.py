# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 09:57:43 2019

@author: albhsu
"""

import warnings
warnings.simplefilter('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import auc, roc_curve, classification_report
from sklearn import metrics

import matplotlib.pyplot as plt


#%matplotlib inline

#Load Dataset
data = pd.read_csv('PG_test.csv')
data.head()
data.info()

#data['y'] = data['PostRR'] - data['CurRR']

#Machine Learning
data['is_train'] = np.random.uniform(0, 1, len(data)) <= .75

train, test = data[data['is_train']==True], data[data['is_train']==False]
features = data.columns[6:67]
#features = data.columns[11:71]

y_train = pd.factorize(train['PGDEF'])[0]

clf = LogisticRegression(random_state=0, solver='lbfgs',multi_class='ovr')
#clf = RandomForestClassifier(n_estimators = 300, n_jobs=2, random_state=0)
clf.fit(train[features], y_train)
y_pred = clf.predict(test[features])

clf.score(test[features], test['PGDEF'])
cm = metrics.confusion_matrix(test['PGDEF'], y_pred)
print(cm)


importance = pd.DataFrame(index = features, data = clf.feature_importances_, columns = ['importance'])
importance.sort_values(by = 'importance', ascending = True, inplace = True)

#Plot feature importance
ax = importance[-5:].plot.barh()

y_pos = np.arange(len(features))
plt.barh(y_pos, clf.coef_.ravel())
plt.show()

y_pred = clf.predict(test[features])
y_actual = test['PGDEF']

#Confusion matrix
pd.crosstab(test['y'], y_pred)

#Classification report
target_names = ['0', '1']
print(classification_report(y_actual, y_pred, target_names=target_names))
fpr, tpr, thresholds = metrics.roc_curve(y_pred, y_actual, pos_label=2)





