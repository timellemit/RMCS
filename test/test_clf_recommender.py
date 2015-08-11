from sklearn.neighbors.classification import KNeighborsClassifier
from sklearn.linear_model.logistic import LogisticRegression
from src.classification_table import classification_table
import numpy as np
from src.classifier_recommender_bfgs import recommend_classifier
from src.preprocess_arff import preprocess
from sklearn.cross_validation import StratifiedKFold

if __name__ == "__main__":
    datasets = [('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
                ('nursery', '../data/dataset_26_nursery.arff'),
                ('diabetes', '../data/dataset_37_diabetes.arff'),
                ('iris', '../data/dataset_61_iris.arff')]
#     
    clfs_and_params = [(KNeighborsClassifier, [{'n_neighbors': k} 
                                               for k in xrange(3, 7)]),
                       (LogisticRegression, [{'C': c} 
                                             for c in 10.0 ** np.arange(-1, 3)])]
#     
    data, target = preprocess(datasets[0][1])
    folds = list(StratifiedKFold(target, 4,
                                 shuffle=True, random_state=42))
    for train_indexes, test_indexes in folds:
        x_train, y_train = data[train_indexes], target[train_indexes]
        x_test, y_test = data[test_indexes], target[test_indexes]
        classification_context = classification_table(x_train, y_train, 
                                                      clfs_and_params)
        print "rec clf", recommend_classifier(x_train, x_test[0], 
                                              classification_context)
        
    
                