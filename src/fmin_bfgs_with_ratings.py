import numpy as np
import scipy.optimize

Theta = np.array([[1,2,1],[2,3,2]])
Y = np.array([[.5, .3],[.2, .8], [.6, .4]])
Y_means = np.mean(Y,axis=1).reshape([3,1])

def func(params, *args):
    Theta = args[0]
    Y = np.hstack((args[1],params[3:].reshape(3,1))) - Y_means.dot(np.ones([1,3]))
    X = np.vstack((np.array([1,1,1]),params[:3])) 
    Y_model = Theta.T.dot(X)
    E = Y-Y_model
    return sum(np.diag(E.T.dot(E)))
 
params0 = np.random.rand(6)*1e-10
result = scipy.optimize.fmin_l_bfgs_b(func, x0=params0,
                                    args=(Theta,Y), approx_grad=True)
learned_x, learned_y = result[0][:3], result[0][3:]
print learned_x, learned_y +Y_means.T

# equivalent for checking
# def handwritten_optim_objective(x1,x2,x3,y1,y2,y3):
#     return (1+2*x1-0.5)**2 + (1+2*x2-0.3)**2 + (1+2*x3-0.6)**2 + \
#         (2+3*x1-0.3)**2 + (2+3*x2-0.8)**2 + (2+3*x3-0.4)**2 + \
#         (3+x1-y1)**2 + (3+x2-y2)**2 + (3+x3-y3)**2

# print handwritten_optim_objective(*result[0]), \
#     handwritten_optim_objective(*params0)