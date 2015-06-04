from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from classification_table_loo import loo_classification_table
import numpy as np
from src.classifier_recommender_bfgs import recommend_classifier
from src.preprocess import preprocess
 
if __name__ == "__main__":
    datasets = [
        ('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
        ('nursery', '../data/dataset_26_nursery.arff'),
        ('diabetes', '../data/dataset_37_diabetes.arff'),
        ('iris', '../data/dataset_61_iris.arff'),
    ]
    
    clfs = [
        (KNeighborsClassifier, [{'n_neighbors': k} for k in xrange(3, 7)]),
        (LogisticRegression, [{'C': c} for c in 10.0 ** np.arange(-1, 3)])]
    
    data, labels, folds = preprocess(datasets[0][1], 4)
    for train_indexes, test_indexes in folds:
        x_train, y_train = data[train_indexes], labels[train_indexes]
        x_test, y_test = data[test_indexes], labels[test_indexes]
        classification_context = loo_classification_table(x_train, y_train, clfs)
        print recommend_classifier(x_train, x_test[0], classification_context)