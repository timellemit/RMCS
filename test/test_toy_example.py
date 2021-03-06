import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from src.RMCS import RMCS

X = np.array([[0, 0], [1, 0], [2, 0],
              [0, 1], [1, 1], [0, 2],
              [3, 0], [2, 1], [3, 1],
              [1, 2], [2, 2], [3, 2],
              ])
y = np.array([0, 0, 0, 0, 0, 0,
              1, 1, 1, 1, 1, 1])

clfs_and_params = [(KNeighborsClassifier, [{'n_neighbors': k} 
                                           for k in xrange(3, 7)]),
                   (LogisticRegression, [{'C': c} 
                                         for c in 10.0 ** np.arange(-1, 3)])]

clf = RMCS(clfs_and_params, fold_fraction=1./4)
clf.fit(X, y) 
print clf.predict(X[0:3,:])
