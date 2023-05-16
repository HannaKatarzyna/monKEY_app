import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler


def filter_record(data, key_filter=False):
    # cut first key -> wrong, confusing times [1:]

    # basic time filters
    df_new = data.drop(data[data['holdTime'] > 5].index, inplace=False)
    df_new.drop(df_new[df_new['holdTime'] < 0].index, inplace=True)
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


# def feature_extract_method_1(data, assumed_length=360, window_time=90, normalize_option=False):


def feature_extract_method_2(data, dynamic_feature='holdTime', time_feature='releaseTime', assumed_length=360, window_time=15, normalize_option=False):

    # number of non-overlapping windows
    n_windows = int(assumed_length/window_time)

    t_inter = np.arange(0, 1.01, 0.01)  # for KDE
    stat_moments = np.zeros([4, n_windows])
    data_for_cov = np.zeros([n_windows, len(t_inter)])

    # fig, axs = plt.subplots(figsize=[4, 3])

    for i in range(n_windows):

        df_temp = data[(data[time_feature] > i*window_time)
                       & (data[time_feature] < (i+1)*window_time)]

        if len(df_temp) < 6:
            print('Not enough samples (', len(df_temp),
                  ') in this window - it has to be omitted')
            continue

        # # to normalize here only for FLIGHT TIME: zero-mean?
        # if normalize_option:
        #     scaler = MinMaxScaler()
        #     # scaler = StandardScaler()
        #     scaler.fit(df_temp[dynamic_feature])
        #     df_temp[dynamic_feature] = scaler.transform(
        #         df_temp[dynamic_feature])

        # first order
        stat_moments[0, i] = df_temp[dynamic_feature].mean()
        # second order
        stat_moments[1, i] = df_temp[dynamic_feature].std()
        # third order
        stat_moments[2, i] = df_temp[dynamic_feature].kurtosis()
        # fourth order
        stat_moments[3, i] = df_temp[dynamic_feature].skew()

        # print(len(df_temp))
        # print(stat_moments[0, i])
        # PDF estimation via KDE
        X = df_temp['holdTime'].to_numpy().reshape(-1, 1)

        bandwidth = np.arange(0.05, 2, .1)
        kde = KernelDensity(kernel='gaussian')
        grid = GridSearchCV(kde, {'bandwidth': bandwidth})
        grid.fit(X)
        kde = grid.best_estimator_
        # print("optimal bandwidth: " + "{:.2f}".format(kde.bandwidth))
        # kde = KernelDensity(kernel='gaussian', bandwidth=1.9).fit(X)
        log_density = np.exp(kde.score_samples(t_inter.reshape(-1, 1)))
        data_for_cov[i, :] = log_density

        # axs.plot(t_inter, log_density)

    va = np.zeros(11)
    for i in range(4):
        va[i*2], va[i*2+1] = np.mean(stat_moments[i, :]
                                     ), np.std(stat_moments[i, :], ddof=1)

    # a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    # # extract values only from upper triangle of matrix
    # upper_triangle = a[np.triu_indices_from(a)]
    # # or above the diagonal
    # upper_triangle = a[np.triu_indices_from(a, k=1)]

    cov_matrix = np.cov(data_for_cov)
    upper_triangle = np.abs(cov_matrix[np.triu_indices_from(cov_matrix)])
    va[8] = np.mean(upper_triangle)
    va[9] = np.std(upper_triangle, ddof=1)
    va[10] = np.sum(upper_triangle)

    return va

# https://scikit-learn.org/stable/modules/density.html#kernel-density-estimation
# https://stackabuse.com/kernel-density-estimation-in-python-using-scikit-learn/
