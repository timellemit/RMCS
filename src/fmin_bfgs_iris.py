import numpy as np
from numpy.random import random, choice
import scipy.optimize
from sklearn import datasets

# numbers of objects and classifiers, and object features
N_OBJ, N_FEAT, N_CLF = 150, 4, 4

iris = datasets.load_iris()

# object features. (N_FEAT + 1) x N_OBJ
X = np.vstack([np.array([1]*iris.data.shape[0]),iris.data.T])
print "X.shape", X.shape
# classifier parameters (N_FEAT + 1) x N_CLF
Theta = np.array([[choice([1,-1])*random()/pow(10,5) 
                  for i in xrange(N_CLF)]
                  for j in xrange(N_FEAT + 1)])
print "Theta.shape", Theta.shape
# scores - randomly between 0.5 and 1
Y = np.array([[0.5*(random() + 1) 
               for j in xrange(N_CLF)]
              for i in xrange(N_OBJ)])

def func(params, *args):
    X = args[0]
    #print "X.shape" , X.shape
    Y = args[1]
    Theta = params
    Theta = np.reshape(Theta, (N_FEAT + 1, N_CLF))
    #print "Theta.shape" , Theta.shape
     
    Y_model = Theta.T.dot(X)
    #print "Y_model", Y_model
    E = Y.T-Y_model
    return sum(np.diag(E.T.dot(E)))
 
resp = scipy.optimize.fmin_l_bfgs_b(func, x0=np.ndarray.flatten(Theta),
                                    args=(X,Y), approx_grad=True)
#print np.reshape(resp[0],(N_FEAT + 1, N_CLF))

print "first iris", X[:,1]
print "classifications for first iris", Y[1,:]
print "classifications for first iris", Y[:,1]
new_iris = np.array([1, 4.9, 3, 1.4, 0.2])
print "new iris", new_iris
learned_theta = np.reshape(resp[0],(N_FEAT + 1, N_CLF)).T
print "classifications for new iris", learned_theta.dot(new_iris)


def predict_classifications(train_set, test_object, clf_table):
    n_obj, n_feat, n_clfs = train_set.shape[0], train_set.shape[1], clf_table.shape[1]
    