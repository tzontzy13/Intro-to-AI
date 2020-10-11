import os
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


wine = datasets.load_wine()

#print(wine.target[:5])
#print(wine.data[:5])
#print(wine.feature_names)

df = pd.DataFrame(data=np.c_[wine['data'],wine['target']],
                  columns=np.append(wine['feature_names'],['target']))

df = df.reindex(np.random.permutation(df.index))

#print(df.head())

result = []
for x in df.columns:
    if x != 'target':
        result.append(x)

X = df[result].values
y = df['target'].values
variety = wine.target_names

kf = KFold(5)

sc = StandardScaler()
tree = DecisionTreeClassifier(criterion='entropy')

#print(df.head())
#print(x)
#print(y)

#using 5-fold cross validation, for each fold, a DT
#is trained and tested and confusion matrices generated
for train_index, validate_index in kf.split(X,y):
    Xt = X[train_index]
    sc.fit(Xt)
    X_train_std = sc.transform(Xt)
    X_test_std = sc.transform(X[validate_index])
    tree.fit(X_train_std,y[train_index])
    y_test = y[validate_index]
    y_pred = tree.predict(X_test_std)
    #print(y_test)
    #print(y_pred)
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    ##
    cm = confusion_matrix(y_test, y_pred)
    np.set_printoptions(precision=3)
    print('Confusion matrix, without normalization')
    print(cm)
    plt.figure()
    #plot_confusion_matrix(cm, variety, title='')
    ##
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('Normalized confusion matrix')
    print(cm_normalized)
    plt.figure()
    #plot_confusion_matrix(cm_normalized, variety, title='Normalized confusion matrix')
    #plt.show()