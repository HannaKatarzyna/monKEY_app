import sys
sys.path.append('../')
from kmodule.keystroke_module import *
import numpy as np

##
X = np.load('nq_ex1_features.npy')
Y = np.load('nq_ex1_gt.npy')

print('  KNN')
cross_validation(X, Y, train_func=train_kNN_model)

print('  SVM')
cross_validation(X, Y, train_func=train_SVM_model)

if len(sys.argv)>1:
    # https://machinelearningmastery.com/train-final-machine-learning-model/
    model = train_kNN_model(X, Y, n_n=3)
    with open('model_' + str(sys.argv[1]) + '.pkl', 'wb') as file:
        pickle.dump(model, file)