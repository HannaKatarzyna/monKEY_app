import sys
sys.path.append('../')
from kmodule.keystroke_module import *
import numpy as np

##
X = np.load('nq_ex1_features.npy')
Y = np.load('nq_ex1_gt.npy')

# print('  KNN')
# cross_validation(X, Y, train_func=train_kNN_model)

print('  SVM')
cross_validation(X, Y, train_func=train_SVM_model)

# if len(sys.argv)>1:
#     # https://machinelearningmastery.com/train-final-machine-learning-model/
#     model = train_kNN_model(X, Y, n_n=3)
#     with open('model_' + str(sys.argv[1]) + '.pkl', 'wb') as file:
#         pickle.dump(model, file)


# scaler = MinMaxScaler()
# X_norm = scaler.fit_transform(X)

# # parameters_grid = {'gamma': ['scale', 'auto', 0.01, 0.1, 1. ]}
# parameters_grid = {'C':[1, 10, 100],'kernel': ['linear','rbf', 'poly', 'sigmoid']}
# data = search_params(X_norm, Y, SVC(), parameters_grid)

# parameters_grid = {'n_neighbors': list(range(3, 12, 2)),'p':[1,2,4]}
# data = search_params(X_norm, Y, KNeighborsClassifier(), parameters_grid)


# df = pd.DataFrame(data.cv_results_)
# df.to_csv("SVM_nq_ex2.csv")

# scaler = MinMaxScaler()
# X_norm = scaler.fit_transform(X)
# X_train, X_test, y_train, y_test = train_test_split(X_norm, Y, test_size=0.3, random_state=2)
# trainer, model = train_architecture(X_train,y_train,max_epoch_train = 5)
# acc_val, rep = test_architecture(trainer, model, X_test, y_test)