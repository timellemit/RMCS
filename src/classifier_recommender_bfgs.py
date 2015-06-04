import numpy as np
from numpy.random import random, choice
import scipy.optimize
from sklearn import datasets
 
def predict_classifications(train_set, test_object, clf_table):
    """
    Predicts classification results for all classifiers and a test object
    given classification results on the training set.
    
    train_set - numpy.ndarray (num_objects x num_features). 
    test_object - numpy.ndarray (1 x num_features). Describes a test object for 
                  which classification results are predicted
    clf_table - numpy.ndarray (num_objects x num_classifiers). Cross-validation
                classifications results for each object and each classifier 
    
    Example:
    
    # X is a training set (4 objects, 8 features)
    X = np.array([[1, 2, 1, 4, 4, 5, 6, 5],
              [3, 1, 3, 5, 6, 7, 7, 5],
              [2, 4, 5, 6, 2, 4, 2, 3],
              [4, 2, 4, 7, 6, 2, 1, 2]])

    # Y is a classification table (4 objects, 4 classifiers).
    # e.g. 0.2 in 1st row and 2nd column means that classifier 2 
    # predicts the correct label of object 1 with probability 0.2
    Y = np.array([[.5, .2, .6, .8],
              [.4, .3, .7, .2], 
              [.3, .5, .6, .2],
              [.7, .8, .5, .3]])
    # A new object is very close to the second object 
    
    test_obj = np.array([3.01, 1.01, 3, 5, 6, 7, 7, 5.01])
    print "classifications for test object", predict_classifications(X, test_obj, Y)
    """
    n_obj, n_feat, n_clfs = train_set.shape[0], train_set.shape[1], clf_table.shape[1]
    clf_table
    # add intercept terms to X
    X = np.vstack([np.array([1]*n_obj), train_set.T])
    # initialize Theta with small random values
    Theta = np.array([[choice([1,-1])*random()/pow(10,5) 
                  for _ in xrange(n_clfs)]
                  for _ in xrange(n_feat + 1)])
    
    # optimization objective for BFGS optimizer
    def func(params, *args):
        X = args[0]
#         print "X.shape" , X.shape
        Y = args[1]
        Theta = params
        Theta = np.reshape(Theta, (n_feat + 1, n_clfs))
#         print "Theta.shape" , Theta.shape
        Y_model = Theta.T.dot(X)
        E = Y.T - Y_model
        return sum(np.diag(E.T.dot(E)))
    
    # BFGS optimizer returns 3 values/ We take care only of the optimal Theta
    # fmin_l_bfgs_b returns it as a long vector
    optimal_theta, _, _ = scipy.optimize.fmin_l_bfgs_b(func, x0=np.ndarray.flatten(Theta),
                                    args=(X,clf_table), approx_grad=True)
    # reshape vector optimal_theta
    learned_theta = np.reshape(optimal_theta,(n_feat + 1, n_clfs)).T
    # we add a bias term 1 to the test object and return prediction 
    return learned_theta.dot(np.hstack([1, test_object]))

def recommend_classifier(train_set, test_object, clf_table):
    """
    Returns the number of the classifier, recommended for test_object
    based on the classification table for the training set.
    
    train_set - numpy.ndarray (num_objects x num_features). 
    test_object - numpy.ndarray (1 x num_features). Describes a test object for 
                  which classification results are predicted
    clf_table - numpy.ndarray (num_objects x num_classifiers). Cross-validation
                classifications results for each object and each classifier
    """
    return np.argmax(predict_classifications(train_set, test_object, clf_table))

if __name__ == "__main__":
    
    iris = datasets.load_iris()
    print iris
    # simulate classification table for iris dataset
    clf_table_iris = np.array([[0.5*(random() + 1) 
                    for j in xrange(5)]
                   for i in xrange(iris.data.shape[0])])
    
    print "first iris", iris.data[0,:]
    print "classifications for first iris", clf_table_iris[0,:]
    
    test_iris = np.array([5.1, 3.5, 1.4, 0.2])
    print "new iris", test_iris
    print "classifications for test iris", predict_classifications(iris.data, test_iris, clf_table_iris)
    print recommend_classifier(iris.data, test_iris, clf_table_iris)