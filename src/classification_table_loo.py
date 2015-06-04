# -*- coding: utf-8 -*-
from itertools import product
import pandas as pd
import numpy as np
from scipy.io.arff import loadarff
from sklearn.cross_validation import LeaveOneOut
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.preprocess import preprocess

def loo_classification_table(train, train_labels, classifiers, metrics_count=4):
    # all combinations of classifiers and theirs parameters
    classifiers_and_params =  [(cl,par) for (cl, params) in classifiers for par in params]
    loo_folds = LeaveOneOut(train.shape[0])

    classification_context = np.zeros((train_labels.shape[0], len(classifiers_and_params)))
    # for all classifiers and their parameters
    for classifier_index, clf_and_params in enumerate(classifiers_and_params):
        Classifier, parameters_configurations = clf_and_params
        print Classifier.__name__
        print classifier_index, clf_and_params
  
        classifier = Classifier(**parameters_configurations)
  
        # predicting class for all the instances
        for train_indexes, test_indexes in loo_folds:
            x_train, y_train = train[train_indexes], train_labels[train_indexes]
            x_test, y_test = train[test_indexes], train_labels[test_indexes]
            classifier.fit(x_train, y_train)
            probabilities = classifier.predict_proba(x_test)
            y_pred = classifier.predict(x_test)
            # context value = probability of true class "minus" max value of probability of other classes
            classification_context[test_indexes, classifier_index] = \
            probabilities[xrange(len(test_indexes)), y_test]
            probabilities[xrange(len(test_indexes)), y_test] = 0.0
            classification_context[test_indexes, classifier_index] -= \
            np.max(probabilities, axis=1)
    return classification_context

if __name__ == '__main__':
    datasets = [
    ('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
    ('nursery', '../data/dataset_26_nursery.arff'),
    ('diabetes', '../data/dataset_37_diabetes.arff'),
    ('iris', '../data/dataset_61_iris.arff'),
    ]
    
    clfs = [
    (KNeighborsClassifier, [{'n_neighbors': k} for k in xrange(3, 7)]),
    (LogisticRegression, [{'C': c} for c in 10.0 ** np.arange(-1, 3)])]#,
#     (SVC, [{'kernel': 'rbf', 'C': c, 'gamma': gamma, 'probability': True} for c, gamma in product(
#         [10.0, 1000.0],  # C
#         [0.1, 0.01],     # gamma
#     )]),
# ]
    data, labels, folds = preprocess(datasets[0][1], 4)
    print loo_classification_table(data, labels, clfs)
    
