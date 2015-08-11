import numpy as np
from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from src.preprocess_arff import preprocess
from src.RMCS import RMCS
from sklearn import cross_validation, datasets

data_sets = [('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
            ('nursery', '../data/dataset_26_nursery.arff'),
            ('diabetes', '../data/dataset_37_diabetes.arff'),
            ('iris', '../data/dataset_61_iris.arff')]
data, target = preprocess(data_sets[0][1])
# iris = datasets.load_iris()
# data, target = iris.data, iris.target
clfs_and_params = [(KNeighborsClassifier, [{'n_neighbors': k} 
                                           for k in xrange(3, 7)]),
                   (LogisticRegression, [{'C': c} 
                                         for c in 10.0 ** np.arange(-1, 3)])]

X_train, X_test, y_train, y_test = cross_validation.train_test_split(
        data, target, test_size=0.4, random_state=0)

for clf_index, clf_params in enumerate(clfs_and_params):
        Classifier, param_configs = clf_params
        for params in param_configs:
            print Classifier.__name__, params  
            clf = Classifier(**params)
            clf.fit(X_train, y_train)
            print clf.score(X_test, y_test) 
        
clf = RMCS(clfs_and_params)
clf.fit(X_train, y_train)
print clf.score(X_test, y_test) 

