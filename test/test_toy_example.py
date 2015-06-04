import numpy as np
from classifier_recommender_bfgs import predict_classifications

# X is a training set (4 objects, 2 features)
X = np.array([[1, 2],
              [3, 1],
              [2, 4],
              [4, 2]])

# Y is a classification table (4 objects, 4 classifiers).
# e.g. 0.2 in 1st row and 2nd column means that classifier 2 
# predicts the correct label of object 1 with probability 0.2
Y = np.array([[.5, .2, .6, .8],
              [.4, .3, .7, .2], 
              [.3, .5, .6, .2],
              [.7, .8, .5, .3]])
# A new object is very close to the second object 
print "second object classifications",     
test_obj = np.array([3.01, 1.01])
print "classifications for test object", predict_classifications(X, test_obj, Y)