import numpy as np
import sys
sys.path.append('../')
from kmodule.keystroke_module import *

path = 'extracted_features/'

if len(sys.argv)>2:
    # train model with best hiperparams
    X = np.load(path+str(sys.argv[1])+'_ex2_features_asym.npy')
    Y = np.load(path+str(sys.argv[1])+'_ex2_gt_asym.npy')
    if str(sys.argv[-1]).lower() == "svm":
        model = train_SVM_model(X, Y, 100, 'linear')
    else:
        model = train_kNN_model(X, Y, 3, 1)

    with open('..\models\model_' + str(sys.argv[2]) + '.pkl', 'wb') as file:
        pickle.dump(model, file)

else:

    ##
    X = np.load(path+str(sys.argv[1])+'_ex1_features.npy')
    Y = np.load(path+str(sys.argv[1])+'_ex1_gt.npy')

    print('\n  SVM for ex1 and '+str(sys.argv[1]))
    cross_validation(X, Y, train_SVM_model, 5, 100, 'rbf', 'auto')

    print('\n  KNN for ex1 and '+str(sys.argv[1]))
    cross_validation(X, Y, train_kNN_model, 5, 5, 1)

    ##
    X = np.load(path+str(sys.argv[1])+'_ex2_features.npy')
    Y = np.load(path+str(sys.argv[1])+'_ex2_gt.npy')

    print('\n  SVM for ex2 and '+str(sys.argv[1]))
    cross_validation(X, Y, train_SVM_model, 5, 100, 'linear')

    print('\n  KNN for ex2 and '+str(sys.argv[1]))
    cross_validation(X, Y, train_kNN_model, 5, 3, 1)

    # ##
    # X = np.load(path+str(sys.argv[1])+'_ex2_features_asym.npy')
    # Y = np.load(path+str(sys.argv[1])+'_ex2_gt_asym.npy')

    # print('\n  SVM for ex2 and '+str(sys.argv[1]))
    # cross_validation(X, Y, train_SVM_model, 5, 100, 'linear')

    # print('\n  KNN for ex2 and '+str(sys.argv[1]))
    # cross_validation(X, Y, train_kNN_model, 5, 3, 1)