import numpy as np
import pandas as pd
import re
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from dateutil import parser as date_parser
from datetime import datetime
import warnings

def is_date_parsing(date_str):
    try:
        return bool(date_parser.parse(date_str))
    except ValueError:
        return False


def count_time_from_0(df):
    s1 = pd.Series(df['Timestamp'].iloc[1:]).reset_index()
    s2 = pd.Series(df['Timestamp'].iloc[:-1]).reset_index()
    time_diff = (s1['Timestamp'] - s2['Timestamp']
                 ).apply(lambda x: x.total_seconds())
    time_diff = pd.concat([pd.Series(0.0), time_diff], ignore_index=True)
    return time_diff.cumsum()


def assign_cols(df):
    cond = df['file_2'].apply(lambda x: isinstance(x, float))

    df_1 = df[cond == True]     # NaNs as 2. file
    df_1.drop(columns=['file_2'], inplace=True)

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
    # df_new.drop(df_new[df_new['releaseTime'] < 0].index, inplace=True)

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


def feature_extract_method_1(data, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=90, normalize_option=False):

    n_features = 6
    # number of non-overlapping windows
    n_windows = int(assumed_length/window_time)
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


def feature_extract_method_2(data_orig, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=15, normalize_option=False):

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
    warnings.filterwarnings('error')

    # fig, axs = plt.subplots(figsize=[4, 3])

    # to normalize here only for FLIGHT TIME: zero-mean
    if normalize_option:
        mea = data_orig[dynamic_feature].mean()
        data[dynamic_feature] -= mea
        # data[dynamic_feature] = data_orig[dynamic_feature].apply(lambda x: x - mea)
        # df_temp[dynamic_feature] = df_copied[dynamic_feature] - mea

    for i in range(n_windows):

        df_temp = data[(data[time_feature] > i*window_time)
                       & (data[time_feature] < (i+1)*window_time)]

        typical_number += len(df_temp)
        if len(df_temp) < 6:
            flag += 1
            # print('Not enough samples (', len(df_temp),
            #       ') in this window - it has to be omitted')
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

        # bandwidth = np.arange(0.05, 2, .1)
        bandwidth = np.array([0.05, 0.85, 1.00, 1.45, 1.75, 1.95])
        kde = KernelDensity(kernel='gaussian')
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(X)
        kde = grid.best_estimator_
        # print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))

        # kde = KernelDensity(kernel='gaussian', bandwidth=b_param).fit(X)
        log_density = np.exp(kde.score_samples(t_inter.reshape(-1, 1)))
        data_for_cov[i, :] = log_density

        # axs.plot(t_inter, log_density)

    va = np.zeros(11)
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

        # mean diff between L and R
        tmpL = data[data['Hand'] == 'L']
        tmpR = data[data['Hand'] == 'R']
        va[10] = abs(tmpL[dynamic_feature].mean() -
                    tmpR[dynamic_feature].mean())  # abs or not - ?

        print('Flag / [% of all]: ', flag, '  ', flag/n_windows)
        # print('All windows: ', n_windows)
        # print('Typical number of keys: ', typical_number)

        if np.count_nonzero(np.isnan(va)) > 0:
            print(va)

    except RuntimeWarning:
        print('Warning was raised as an exception!')
        warn_flag = 1

    return va, warn_flag
# https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
# https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/


class nqDataset:
    def __init__(self, filename1, filename2):

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

        self.trainset = None
        self.testset = None
        self.train_ground_truth = None
        self.test_ground_truth = None
        self.features = None

    def show_stats(self):
        print('Patients with PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 1.0]))
        print('Patients without PD: ', len(
            self.user_info[self.user_info['Parkinsons'] == 0.0]))
        plt.figure(figsize=[4, 3])
        sns.countplot(x='Parkinsons', data=self.user_info)
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
            n_features = 22
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
                va_HT, typi = feature_extract_method_2(
                    df_ID, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=20)
                va_NFT, _ = feature_extract_method_2(
                    df_ID, dynamic_feature='flightTime', time_feature='releaseTime', assumed_length=360, window_time=20, normalize_option=True)
                self.features[i, :] = np.concatenate([va_HT, va_NFT], axis=0)

            # if typi < 200:
            #     print('Special case - number of keys: ', typi)

        # shuffle + random_state
        self.trainset, self.testset, self.train_ground_truth, self.test_ground_truth = train_test_split(
            self.features, self.user_info['Parkinsons'].to_numpy(), test_size=0.3, shuffle=True, random_state=42)
