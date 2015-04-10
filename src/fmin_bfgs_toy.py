import numpy as np
from numpy.random import random, choice
import scipy.optimize

# numbers of objects and classifiers, and object features
N_OBJ, N_FEAT, N_CLF = 5, 8, 4

# object features. (N_FEAT + 1) x N_OBJ
X = np.array([[1, 1, 1, 1],
              [1, 3, 2, 4],
              [2, 1, 4, 2],
              [1, 3, 5, 4],
              [4, 5, 6, 7],
              [4, 6, 2, 6],
              [5, 7, 4, 2],
              [6, 7, 2, 1],
              [5, 5, 3, 2],
              ])

# classifier parameters (N_FEAT + 1) x N_CLF
Theta = np.array([[choice([1,-1])*random()/pow(10,5) 
                  for i in xrange(N_CLF)]
                  for j in xrange(N_FEAT + 1)])
# scores
Y = np.array([[.5, .2, .6, .8],
              [.4, .3, .7, .2], 
              [.3, .5, .6, .2],
              [.7, .8, .5, .3]])

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

new_object = np.array([1, 3.01, 1.01, 3, 5, 6, 7, 7, 5.01])
learned_theta = np.reshape(resp[0],(N_FEAT + 1, N_CLF)).T
print learned_theta.dot(new_object)