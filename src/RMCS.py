import numpy as np
from src.classification_table import classification_table
from src.classifier_recommender_bfgs import recommend_classifier

from sklearn.base import BaseEstimator, ClassifierMixin

def classifier_by_num(clfs_and_params, clf_num):
        """
        
        :param clfs_and_params - a list of classifier-param_configs tuples
        example: [(KNeighborsClassifier, [{'n_neighbors': k} for k in xrange(3, 7)]),
            (LogisticRegression, [{'C': c} for c in 10.0 ** np.arange(-1, 3)])]
        :param clf_num - int, an index of the 'best' classifier and param config pair
        returns: an instance of scikit classifier with initialized params 
        """
        running_clf_params_num = 0
        for (clf, param_configs) in clfs_and_params:
            for param_config in param_configs:           
                if running_clf_params_num == clf_num:
                    return clf(**param_config)
                else:
                    running_clf_params_num += 1
                    
class RMCS(BaseEstimator, ClassifierMixin):
    
    def __init__(self, classifiers_and_params, fold_fraction=1./3):
        self.classifiers_and_params = classifiers_and_params
        self.fold_fraction = fold_fraction
        
    def get_params(self, deep=True):
        return BaseEstimator.get_params(self, deep=deep)
    
    def set_params(self, **params):
        return BaseEstimator.set_params(self, **params)
    
    def fit(self, X, y, verbose=False):
        self.X = X
        self.y = y
        self.clf_context = classification_table(X, y, self.classifiers_and_params,
                        fold_num=round(self.X.shape[0]*self.fold_fraction))
        self.trained_clfs = []
        for (clf_name, param_configs) in self.classifiers_and_params:
            for params in param_configs:
                trained_clf = clf_name(**params)
                trained_clf.fit(X, y)
                self.trained_clfs.append(trained_clf)
        
                
    def predict(self, X):
        if len(X.shape) == 1: # only one test object
            X = [X]
#         # we train classifiers lazily 
#         trained_clfs = {}
        predictions = np.array([])
        for test_obj in X:
            # recommended classifier index
            rec_clf_num = recommend_classifier(self.X, test_obj, 
                                           self.clf_context)
#             if not rec_clf_num in trained_clfs.keys():
#                 # recommended classifier (an instance)
#                 rec_clf = classifier_by_num(self.classifiers_and_params,
#                                                 rec_clf_num) 
#                 # train it
#                 rec_clf.fit(self.X, self.y)
#                 trained_clfs[rec_clf_num] = rec_clf
            # make a prediction
            predicted_label = self.trained_clfs[rec_clf_num].predict(test_obj)
            predictions = np.concatenate([predictions,
                                          predicted_label])
        return predictions

if __name__ == "__main__":
    from src.preprocess_arff import preprocess
    from sklearn.neighbors.classification import KNeighborsClassifier
    from sklearn.linear_model.logistic import LogisticRegression
    
    datasets = [
        ('breast_cancer', '../data/dataset_13_breast-cancer.arff'),
        ('nursery', '../data/dataset_26_nursery.arff'),
        ('diabetes', '../data/dataset_37_diabetes.arff'),
        ('iris', '../data/dataset_61_iris.arff')]
  
    clfs_and_params = [(KNeighborsClassifier, [{'n_neighbors': k}
                                               for k in xrange(3, 7)]),
                       (LogisticRegression, [{'C': c}
                              for c in 10.0 ** np.arange(-1, 3)])]
 
    data, target = preprocess(datasets[0][1])
    clf = RMCS(clfs_and_params, train_folds=25)
    clf.fit(data, target)
    print clf.predict(data[0,:])