import os
import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity, KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, train_test_split, KFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.pipeline import make_pipeline
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from dateutil import parser as date_parser
from datetime import datetime
import warnings
import pickle
from torch.utils.data import DataLoader, TensorDataset
from torch import cuda
from torch import Tensor
import torch
from kmodule.myMLP import MLP
import pytorch_lightning as pl
from sklearn import metrics as mt

np.random.seed(42)


def is_date_parsing(date_str):
    try:
        return bool(date_parser.parse(date_str))
    except ValueError:
        return False


def count_time_from_0(df):
    s1 = pd.Series(df['Timestamp'].iloc[1:]).reset_index(drop=True)
    s2 = pd.Series(df['Timestamp'].iloc[:-1]).reset_index(drop=True)
    time_diff = (s1 - s2).apply(lambda x: x.total_seconds())
    time_diff = pd.concat([pd.Series(0.0), time_diff], ignore_index=True)
    return time_diff.cumsum()


# sztuczne powiększenie zbioru nqDataset
def assign_cols(df):
    cond = df['file_2'].apply(lambda x: isinstance(x, float))

    df_1 = df[cond == True]     # NaNs as 2. file
    df_1 = df_1.drop(columns=['file_2'])

    df_2 = df[cond == False]    # 2 files
    df_3 = df_2.drop(columns=['file_2'], inplace=False)
    df_4 = df_2.drop(columns=['file_1'], inplace=False)
    df_4.rename(columns={"file_2": "file_1"}, inplace=True)

    df_conc = pd.concat([df_1, df_3, df_4], ignore_index=True)
    return df_conc


def filter_record(data, key_filter=False):
    # cut first key -> wrong, confusing times [1:]

    # basic time filters
    df_new = data.drop(data[data['holdTime'] > 1].index, inplace=False)
    df_new.drop(df_new[df_new['holdTime'] < 0].index, inplace=True)
    df_new.drop(df_new[df_new['flightTime'] > 3].index, inplace=True)

    if key_filter:
        pattern = 'mouse|BackSpace|Shift|Alt|Control|Num_Lock|Return|P_Enter|Caps_Lock|Left|Right|Up|Down'
        pattern += '|more|less|exclamdown|\[65027\]|\[65105\]|ntilde|minus|equal|bracket|semicolon|slash|apostrophe|grave|question|right|left'
        df_new = df_new[~df_new['pressedKey'].str.contains(
            pattern, regex=True, na=False)]

    return df_new

# re.compile()
# spację, comma i period zostawiamy
# len(df_ID_new[df_ID_new['pressedKey'].str.contains(pattern, regex=True)])/1750
# jak dużo danych tracimy w wyniku filtracji? ~10%


def which_hand(key):
    patternL = 'q|w|e|r|t|a|s|d|f|g|z|x|c|v|b'
    patternR = 'y|u|i|o|p|h|j|k|l|n|m|comma|period|semicolon|slash'
    if key == 'space':
        return 'S'
    elif re.match(patternL, str(key)):
        return 'L'
    elif re.match(patternR, str(key)):
        return 'R'
    else:
        return 'N'


def check_asymmetry(data, dynamic_feature):
    # mean diff between L and R
    tmpL = data[data['Hand'] == 'L']
    tmpR = data[data['Hand'] == 'R']
    var = tmpL[dynamic_feature].mean() - tmpR[dynamic_feature].mean()
    return var


def sampling_imbalanced_data(X, y, opt='under'):
    if str(opt) == 'under':
        print('UNDERSAMPLING')
        rus = RandomUnderSampler(random_state=None)
        X_resampl, y_resampl = rus.fit_resample(X, y)
    else:
        print('OVERSAMPLING')
        ros = RandomOverSampler(random_state=None)
        X_resampl, y_resampl = ros.fit_resample(X, y)

    return X_resampl, y_resampl


def feature_extract_method_1(data, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=90, normalize_option=False):

    n_features = 6

    n_windows = int(data[time_feature].iloc[-1] /
                    window_time)  # wersja dla Tappy
    warnings.simplefilter('ignore', RuntimeWarning)

    # # number of non-overlapping windows - version for NQ
    # n_windows = int(assumed_length/window_time)

    va = np.zeros([n_windows, n_features])

    for i in range(n_windows):
        df_temp = data[(data[time_feature] > i*window_time)
                       & (data[time_feature] < (i+1)*window_time)]

        temp = df_temp[dynamic_feature]
        temp = temp[~np.isnan(temp)]
        Q1 = temp.quantile(q=0.25)
        Q2 = temp.quantile(q=0.5)
        Q3 = temp.quantile(q=0.75)
        IQR = Q3 - Q1
        upper_lim = Q3 + 1.5*IQR
        lower_lim = Q1 - 1.5*IQR
        vout = len(temp[(temp < lower_lim) | (temp > upper_lim)])
        viqr = (Q2 - Q1)/(Q3 - Q1)
        hist, bin_edges = np.histogram(
            temp, bins=4, density=True, range=(0.0, 0.5))
        vhist1, vhist2, vhist3, vhist4 = hist * np.diff(bin_edges)
        va[i, :] = np.array([vout, viqr, vhist1, vhist2, vhist3, vhist4])

    return np.nanmean(va, axis=0)


def feature_extract_method_2(data_orig, n_features=11, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=15, normalize_option=False):

    # number of non-overlapping windows
    # n_windows = int(assumed_length/window_time)
    n_windows = int(data_orig[time_feature].iloc[-1]/window_time)

    t_inter = np.arange(0, 1.01, 0.01)  # for KDE
    stat_moments = np.empty((4, n_windows))
    stat_moments.fill(np.nan)

    # data_for_cov = np.zeros([n_windows, len(t_inter)])
    data_for_cov = np.empty((n_windows, len(t_inter)))
    data_for_cov.fill(np.nan)

    typical_number = 0
    flag = 0

    data = data_orig.copy()

    warn_flag = 0
    warnings.simplefilter('error', RuntimeWarning)

    # fig, axs = plt.subplots(figsize=[4, 3])

    # to normalize here only for FLIGHT TIME: zero-mean
    if normalize_option:
        mea = data_orig[dynamic_feature].mean()
        data[dynamic_feature] -= mea

    for i in range(n_windows):

        df_temp = data[(data[time_feature] > i*window_time)
                       & (data[time_feature] < (i+1)*window_time)]

        typical_number += len(df_temp)
        if len(df_temp) < 6:
            flag += 1
            continue

        # first order
        stat_moments[0, i] = df_temp[dynamic_feature].mean()
        # second order
        stat_moments[1, i] = df_temp[dynamic_feature].std()
        # third order
        stat_moments[2, i] = df_temp[dynamic_feature].kurtosis()
        # fourth order
        stat_moments[3, i] = df_temp[dynamic_feature].skew()

        # TO DO: use params form article insetad of Gridsearch - ?
        # resulting in b = 0.0060, 0.0289, and 0.0300 for HT, NFT, and NP data respectively

        # PDF estimation via KDE
        X = df_temp[time_feature].to_numpy().reshape(-1, 1)

        bandwidth = np.array([0.85, 1.00, 1.45, 1.75, 1.95])
        kde = KernelDensity(kernel='gaussian')
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(X)
        kde = grid.best_estimator_
        # print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))

        # kde = KernelDensity(kernel='gaussian', bandwidth=b_param).fit(X)
        log_density = np.exp(kde.score_samples(t_inter.reshape(-1, 1)))
        data_for_cov[i, :] = log_density

        # axs.plot(t_inter, log_density)

    va = np.zeros(n_features)
    try:
        for i in range(4):
            va[i*2], va[i*2+1] = np.nanmean(stat_moments[i, :]
                                            ), np.nanstd(stat_moments[i, :], ddof=1)

        # # extract values only from upper triangle of matrix
        # upper_triangle = a[np.triu_indices_from(a)]
        # # or above the diagonal
        # upper_triangle = a[np.triu_indices_from(a, k=1)]

        dfc = data_for_cov[~np.isnan(data_for_cov).any(axis=1)]
        cov_matrix = np.cov(dfc)
        upper_triangle = np.abs(cov_matrix[np.triu_indices_from(cov_matrix)])
        va[8] = np.mean(upper_triangle)
        va[9] = np.std(upper_triangle, ddof=1)
        va[10] = np.sum(upper_triangle)

        if n_features == 12:
            va[11] = check_asymmetry(data, dynamic_feature)

        with open('bandwidth_18_12.txt', 'a') as log_file:
            log_file.writelines(str(np.round(kde.bandwidth, 2)) + '\n')

    except RuntimeWarning:
        print('Warning was raised as an exception!')
        warn_flag = 1

    if n_windows - flag < int(assumed_length/window_time):  # 24
        warn_flag = 1

    return va, warn_flag
# https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
# https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/


def search_params(X, Y, model, param_grid):
    # param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001],'kernel': ['rbf', 'poly', 'sigmoid']}
    # grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
    grid = GridSearchCV(model, param_grid, refit=True)
    grid.fit(X, Y)
    print(grid.best_params_)
    return grid


def cross_validation(X, Y, train_func, n_splits, *args):

    k_folds = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    acc_scores = []

    for k, (train, test) in enumerate(k_folds.split(X)):
        print('Number of fold: ', k+1)

        X_train = X[train, :]
        Y_train = Y[train]
        X_test = X[test, :]
        Y_test = Y[test]

        X_train, Y_train = sampling_imbalanced_data(
            X[train, :], Y[train], opt='under')

        model = train_func(X_train, Y_train, *args)
        acc_val, df_rep = test_selected_model(X_test, Y_test, model)
        acc_scores.append(acc_val)

        if k == 0:
            df_meas = df_rep
        else:
            df_meas = pd.concat([df_meas, df_rep],
                                axis=0, ignore_index=True)

    print(df_meas.mean())
    acc_scores = [round(elem, 2) for elem in acc_scores]
    print("Cross Validation Accuracy Scores: ", acc_scores)
    print("Average CV Score: ", np.mean(acc_scores))


def train_SVM_model(x, y, *args):

    if len(args) > 2:
        clf = make_pipeline(MinMaxScaler(), SVC(
            C=args[0], kernel=args[1], gamma=args[2]))
    else:
        clf = make_pipeline(MinMaxScaler(), SVC(C=args[0], kernel=args[1]))
    clf.fit(x, y)
    return clf


def train_kNN_model(x, y, *args):

    clf = make_pipeline(
        MinMaxScaler(), KNeighborsClassifier(n_neighbors=args[0], p=args[1]))
    clf.fit(x, y)
    return clf


# https://stackoverflow.com/questions/35363530/python-scikit-learn-mlpclassifier-hidden-layer-sizes
def train_MLP_model(x, y, lr=0.01, max_it=500):

    clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(
        x.shape[0]-1, int(x.shape[0]/2), 3, ), activation='relu', solver='adam',
        batch_size='auto', learning_rate='adaptive', learning_rate_init=lr,
        random_state=2, max_iter=max_it))
    clf.fit(x, y)
    return clf


def test_selected_model(x_test, Y_test, model):
    predictions = model.predict(x_test)
    acc_val = accuracy_score(Y_test, predictions)
    rep = precision_recall_fscore_support(
        Y_test, predictions, average='binary', pos_label=1)
    df_rep = pd.DataFrame(
        [rep], columns=['precision', 'recall', 'fscore', 'support'])
    return acc_val, df_rep


def train_architecture(X, Y, n_features, max_epoch_train=50):

    training_data = TensorDataset(Tensor(X), Tensor(Y))

    mo_mlp = MLP(n_features)
    if cuda.is_available():
        train_dataloader = DataLoader(
            training_data, batch_size=8, shuffle=True)
        trainer = pl.Trainer(accelerator='gpu', max_epochs=max_epoch_train)
    else:
        train_dataloader = DataLoader(
            training_data, batch_size=1, shuffle=True)
        trainer = pl.Trainer(max_epochs=max_epoch_train)

    trainer.fit(mo_mlp, train_dataloader)
    mo_mlp.plotting()
    return trainer, mo_mlp


def test_architecture(trainer, model, X, Y):

    test_data = TensorDataset(Tensor(X), Tensor(Y))
    test_dataloader = DataLoader(test_data, batch_size=8, shuffle=False)
    trainer.test(dataloaders=test_dataloader)
    preds_ls = trainer.predict(model, test_dataloader)
    preds = [np.round(np.squeeze(el.numpy())) for el in preds_ls]
    preds = np.concatenate(preds)
    acc_val = accuracy_score(Y, preds)
    rep = precision_recall_fscore_support(
        Y, preds, average='binary', pos_label=1)
    df_rep = pd.DataFrame(
        [rep], columns=['precision', 'recall', 'fscore', 'support'])
    return acc_val, df_rep


class nqDataset:
    def __init__(self, filename1, filename2, st_asym=0):

        # load data
        df1 = pd.read_csv(filename1)
        df2 = pd.read_csv(filename2)
        df_conc = pd.concat([df1, df2], ignore_index=True)
        df_conc.rename(columns={"updrs108": "updrs",
                       "gt": "Parkinsons"}, inplace=True)
        df_conc.drop('nqScore', axis=1, inplace=True)
        df_conc['Parkinsons'] = df_conc['Parkinsons'].map(
            {True: 1.0, False: 0.0})

        # self.user_info = df_conc
        self.user_info = assign_cols(df_conc)
        self.features = None
        self.ground_truth = self.user_info['Parkinsons'].to_numpy()
        self.state_asymmetry = st_asym

    def show_stats(self):
        print('Patients with PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 1.0]))
        print('Patients without PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 0.0]))
        plt.figure(figsize=[4, 3])
        sns.countplot(x='Parkinsons', hue='Parkinsons', data=self.user_info,
                      legend=False, palette=['#432371', "#FAAE7B"])
        plt.xlabel('Parkinson\'s disease')

    @staticmethod
    def load_record(filename):
        df = pd.read_csv(filename, header=None, names=[
            'pressedKey', 'holdTime', 'releaseTime', 'pressTime'])
        df['flightTime'] = df['pressTime'] - \
            pd.concat([pd.Series(0.0), df['releaseTime']], ignore_index=True)
        df['latencyTime'] = df['flightTime'] + \
            pd.concat([pd.Series(0.0), df['holdTime']], ignore_index=True)
        df['Hand'] = df['pressedKey'].apply(which_hand)

        return df

    def prepare_dataset(self, path, feature_extract=2):

        if feature_extract == 2:
            n_features = 22 + 2*self.state_asymmetry
        else:
            n_features = 6

        self.features = np.zeros([len(self.user_info), n_features])

        # DO NOT iterate: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
        # use .apply() instead

        for i, row in self.user_info.iterrows():
            print(i)
            df_ID = nqDataset.load_record(path + row['file_1'])
            df_ID = filter_record(df_ID, key_filter=True)

            if feature_extract == 1:
                va_HT = feature_extract_method_1(
                    df_ID, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=90)
                self.features[i, :] = va_HT

            if feature_extract == 2:
                va_HT, _ = feature_extract_method_2(
                    df_ID, n_features=int(n_features/2), dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=20)
                va_NFT, _ = feature_extract_method_2(
                    df_ID, n_features=int(n_features/2), dynamic_feature='flightTime', time_feature='releaseTime', assumed_length=360, window_time=20, normalize_option=True)
                self.features[i, :] = np.concatenate([va_HT, va_NFT], axis=0)

            # if typi < 200:
            #     print('Special case - number of keys: ', typi)


class tappyDataset:

    def __init__(self, path1, path2, opt=1):

        # load data
        self.files_list1 = os.listdir(path1)
        self.files_list2 = os.listdir(path2)
        users = [el.split("_")[1].split(".")[0] for el in self.files_list1]
        df = pd.DataFrame(data=users, columns=['pID'])

        if opt == 1:
            with open('reports_18_12.txt', 'r') as f:
                lsID = f.readlines()
            lsID = '|'.join(lsID)
            clean_lsID = lsID.replace('\n', '')
            df.drop(df[df['pID'].str.contains(clean_lsID)].index, inplace=True)
            df.reset_index(inplace=True)

        for i, el in enumerate(df['pID']):
            temp = [x for x in self.files_list2 if x.startswith(el)]
            if len(temp):
                df.loc[[i], 'files'] = pd.Series([temp], index=df.index[[i]])
                with open(path1 + self.files_list1[i], 'r') as f:
                    for line in f:
                        if 'Parkinsons' in line:
                            read_info = line.strip().split(": ")
                            df.loc[[i], 'Parkinsons'] = read_info[1]

        df.dropna(inplace=True)
        df['Parkinsons'] = df['Parkinsons'].map({'True': 1.0, 'False': 0.0})

        self.user_info = df

        self.features = None
        self.ground_truth = self.user_info['Parkinsons'].to_numpy()
        self.flag_fatal = []

    def show_stats(self):
        print('Patients with PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 1.0]))
        print('Patients without PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 0.0]))
        plt.figure(figsize=[4, 3])
        sns.countplot(x='Parkinsons', hue='Parkinsons', data=self.user_info,
                      legend=False, palette=['#432371', "#FAAE7B"])
        plt.xlabel('Parkinson\'s disease')

    # TO DO:
    @staticmethod
    def load_record(filename):
        df = pd.read_csv(filename, delimiter="\t", index_col=False, header=None, names=[
                         'User', 'Date', 'Timestamp', 'Hand', 'holdTime', 'Direction', 'flightTime', 'latencyTime'], low_memory=False)

        df.drop(columns=['User'], inplace=True)
        df.drop(columns=['Direction'], inplace=True)

        df[df.columns[-3:]] = df[df.columns[-3:]
                                 ].apply(lambda x: x.lstrip('0') if type(x) == str else x)
        df[df.columns[-3:]
           ] = pd.to_numeric(df[df.columns[-3:]].stack(), errors='coerce').unstack()
        df[df.columns[-3:]] = df[df.columns[-3:]].apply(lambda x: x/1000)

        return df

    @staticmethod
    def grouping_date(df):
        grouped_data = df.groupby('Date').agg(list)
        grouped_data['Length'] = grouped_data['Timestamp'].apply(len)

        idx = grouped_data['Length'].idxmax()
        ex_rec = df[df['Date'] == idx].copy()
        temp = ex_rec['Timestamp'].apply(lambda x: is_date_parsing(x))
        ex_rec.drop(ex_rec[temp == False].index, inplace=True)

        ex_rec.reset_index(inplace=True)
        ex_rec['Timestamp'] = ex_rec['Timestamp'].apply(
            lambda x: datetime.strptime(x, '%H:%M:%S.%f'))
        ex_rec['timeLapse'] = count_time_from_0(ex_rec)

        # indices_nan = np.where(ex_rec['timeLapse'].isna())
        # if len(indices_nan[0]) > 1:
        #     print('Nans:', indices_nan)

        return ex_rec

    def loc_prep(self, i, df_ID, feature_extract):

        if feature_extract == 1:
            va_HT = feature_extract_method_1(
                df_ID, dynamic_feature='holdTime', time_feature='timeLapse', assumed_length=360, window_time=15)
            self.features[i, :] = va_HT
            print(va_HT)
            warn_flag = 0

        if feature_extract == 2:
            va_HT, warn_flag = feature_extract_method_2(
                df_ID, n_features=11, dynamic_feature='holdTime', time_feature='timeLapse', assumed_length=360, window_time=15)
            va_NFT, warn_flag = feature_extract_method_2(
                df_ID, n_features=11, dynamic_feature='flightTime', time_feature='timeLapse', assumed_length=360, window_time=15, normalize_option=True)
            self.features[i, :] = np.concatenate([va_HT, va_NFT], axis=0)

        return warn_flag

    def prepare_dataset(self, path, feature_extract=2):

        if feature_extract == 2:
            n_features = 22
        else:
            n_features = 6

        self.features = np.zeros([len(self.user_info), n_features])

        for i, row in self.user_info.reset_index(inplace=False).iterrows():
            # for i, row in self.user_info.iterrows():
            # for i, row in self.train_df.reset_index(inplace=False).iterrows():
            # .iloc[45:]
            # DO NOT iterate: https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas
            # use .apply() instead

            # control
            print('\nIndex: ', i)
            counter = 0

            print('L: ', len(row['files']))
            while counter < len(row['files']):

                filename = path + row['files'][counter]
                df_ID_all = tappyDataset.load_record(filename)
                df_ID = tappyDataset.grouping_date(df_ID_all)
                df_ID = filter_record(df_ID, key_filter=False)
                warn_flag = self.loc_prep(i, df_ID, feature_extract)

                if warn_flag == 0:
                    print('used')
                    break

                if warn_flag == 1 and counter == len(row['files']) - 1:

                    print('Record not useful, with gt: ', row['Parkinsons'])
                    print('Counter: ', counter)
                    self.flag_fatal.append(i)
                    with open('reports_xx_xx.txt', 'a') as log_file:
                        log_file.writelines(row['pID'] + '\n')

                counter += 1

        print('     SUCCESS!!!')
