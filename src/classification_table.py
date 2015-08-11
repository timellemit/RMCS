# -*- coding: utf-8 -*-
import numpy as np
from sklearn.cross_validation import StratifiedKFold

SEED = 42

def classification_table(train, train_labels, classifiers, 
                         fold_num=4, verbose=False):
    # all combinations of classifiers and theirs parameters
    classifiers_and_params =  [(cl,par) for (cl, params) in classifiers 
                               for par in params]
    folds = list(StratifiedKFold(train_labels, fold_num,
                                 shuffle=True, random_state=SEED))
    clf_context = np.zeros((train_labels.shape[0], len(classifiers_and_params)))
    # for all classifiers and their parameters
    for classifier_index, clf_and_params in enumerate(classifiers_and_params):
        Classifier, parameters_configurations = clf_and_params
        if verbose:
            print Classifier.__name__
            print classifier_index, clf_and_params
  
        classifier = Classifier(**parameters_configurations)
  
        # predicting class for all the instances
        for train_indexes, test_indexes in folds:
            x_train, y_train = train[train_indexes], train_labels[train_indexes]
            x_test, y_test = train[test_indexes], train_labels[test_indexes]
            classifier.fit(x_train, y_train)
            probabilities = classifier.predict_proba(x_test)
            clf_context[test_indexes, classifier_index] = \
            probabilities[xrange(len(test_indexes)), y_test]
            probabilities[xrange(len(test_indexes)), y_test] = 0.0
            clf_context[test_indexes, classifier_index] -= \
            np.max(probabilities, axis=1)
    return clf_context

if __name__ == '__main__':
    from src.preprocess_arff import preprocess
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    
    datasets = [('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
                ('nursery', '../data/dataset_26_nursery.arff'),
                ('diabetes', '../data/dataset_37_diabetes.arff'),
                ('iris', '../data/dataset_61_iris.arff')]
    
    clfs_and_params = [(KNeighborsClassifier, [{'n_neighbors': k}
                                    for k in xrange(3, 7)]),
                       (LogisticRegression, [{'C': c}
                                  for c in 10.0 ** np.arange(-1, 3)])]
    
    data, target = preprocess(datasets[0][1])
    print classification_table(data, target, clfs_and_params, fold_num=4)
    
