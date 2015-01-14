import numpy as np
import scipy.optimize

Theta = np.array([[1,1,2],[1,2,3],[1,3,1]])
Y = np.array([[.5, .2, .6],[.3, .8, .4]])

def func(params, *args):
    Theta = args[0]
    Y = args[1]
    X = params
    X = np.reshape(X, (3,2))
     
    Y_model = X.T.dot(Theta)
    E = Y-Y_model
    return sum(np.diag(E.T.dot(E)))
 
X0 = np.zeros((3, 2))
resp = scipy.optimize.fmin_l_bfgs_b(func, x0=np.ndarray.flatten(X0),
                                    args=(Theta,Y), approx_grad=True)
print np.reshape(resp[0],(3,2))
