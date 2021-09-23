import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import config
import utility
import dataset_generator
import regressors_lib
from utility import get_square_number_array
import statistic_utility

NAMES = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree"]
CLASSIFIERS = [
    RandomForestRegressor(),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    DecisionTreeRegressor(random_state=0)
]

CLASSIFIERS_DICT = {
    "Random forest": RandomForestRegressor(),
    "Nearest Neighbors U": KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    "Nearest Neighbors D": KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    "Decision Tree": DecisionTreeRegressor(random_state=0)
}


def performance_dataset(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.20)

    lox, loy = regressors_lib.get_optimal_points(y_test)

    errors = {}
    for name, clf in zip(NAMES, CLASSIFIERS):
        lpx, lpy = regressors_lib.get_regressor_predicted_points(clf, x_train, x_test, y_train)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        errors[f'{name}--IQR-X'] = iqr(error_x)
        errors[f'{name}--IQR-Y'] = iqr(error_y)
        errors[f'{name}--MEDIAN X'] = np.median(error_x)
        errors[f'{name}--MEDIAN Y'] = np.median(error_y)

    return errors


def get_number_of_good_point(x_train, x_test, y_train, y_test):
    lox, loy = regressors_lib.get_optimal_points(y_test)

    good_points = {}
    for name, clf in zip(NAMES, CLASSIFIERS):
        lpx, lpy = regressors_lib.get_regressor_predicted_points(clf, x_train, x_test, y_train)

        good_point = 0
        for i in range(len(lpx)):
            p_predicted = np.array(lpx[i], lpy[i])
            p_optimal = np.array(lox[i], loy[i])
            dist = np.linalg.norm(p_predicted - p_optimal)
            if dist < config.OPTIMAL_TH:
                good_point += 1

        good_points[name] = good_point

    return good_points


def plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="default"):
    lox, loy = regressors_lib.get_optimal_points(y_test)

    error_x_df = pd.DataFrame()
    error_y_df = pd.DataFrame()
    for name, clf in zip(NAMES, CLASSIFIERS):
        lpx, lpy = regressors_lib.get_regressor_predicted_points(clf, x_train, x_test, y_train)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        error_x_df.insert(len(error_x_df.columns), name, error_x)
        error_y_df.insert(len(error_y_df.columns), name, error_y)

        df = pd.DataFrame({
            'diff x': error_x,
            'diff y': error_y
        })

        df.plot.kde(title=f'{name} {title_add}')

    error_x_df.plot.box(title=f"Error x {title_add}")
    error_y_df.plot.box(title=f"Error y {title_add}")


def get_ecdf_regressor_dataset(x_train, x_test, y_train, y_test, regressor, regressor_name):
    optimal_points, predicted_points = regressors_lib.get_regressor_optimal_and_predicted_points(regressor, x_train,
                                                                                                 x_test, y_train,
                                                                                                 y_test)
    return statistic_utility.get_ecdf_euclidean_df(optimal_points, predicted_points, regressor_name)


def get_ecdf_dataset_squares(x_train, x_test, y_train, y_test, regressors, regressor_name):
    lox, loy = regressors_lib.get_optimal_points(y_test)

    regressors.fit(x_train, y_train)
    Z = regressors.predict(x_test)

    lpr = Z[:, 0]
    lpp = Z[:, 1]
    lpx, lpy = utility.pol2cart(lpr, lpp)

    xo, yo = get_square_number_array(lox, loy)
    xp, yp = get_square_number_array(lpx, lpy)

    df = statistic_utility.get_ecdf_square_df(xo, yo, xp, yp, regressor_name)

    return df


def main_test_dataset_default_vs_dataset_mean_and_std():
    name_file_reader = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    name_file_cam = ["Cal3105run0", "Cal3105run1", "Cal3105run2"]

    x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")

    x_test, y_test = dataset_generator.generate_dataset(name_file_reader, name_file_cam,
                                                        dataset_generator.generate_dataset_base)

    plot_test_multi_regress(x_train, x_test, y_train, y_test)

    x_train, y_train = dataset_generator.generate_dataset_with_mean_and_std("BLE2605r", "2605r0")

    x_test, y_test = dataset_generator.generate_dataset(name_file_reader, name_file_cam,
                                                        dataset_generator.generate_dataset_with_mean_and_std)

    plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="mean_std")


def compare_experiment_with_ecdf():
    """
    Compare the performance of a regressor with different runs by ecdf
    :return: void
    """
    name_files_reader = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    name_files_cam = ["Cal3105run0", "Cal3105run1", "Cal3105run2"]

    x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")

    ecdf_total = pd.DataFrame()
    for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
        x_test, y_test = dataset_generator.generate_dataset([name_file_reader], [name_file_cam],
                                                            dataset_generator.generate_dataset_base)

        # regressor_dict = {
        #     f'kNN_{name_file_reader}': KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance')
        # }
        #
        # ecdf_df = get_ecdf_dataset_back(x_train, x_test, y_train, y_test, regressor_dict)
        ecdf_df = get_ecdf_regressor_dataset(x_train, x_test, y_train, y_test,
                                             KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
                                             f'kNN_{name_file_reader}')
        ecdf_total = pd.concat([ecdf_total, ecdf_df], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')
    ecdf_total.plot.line(
        title="ECDF per different Run",
        xlabel="(m)",
        ylabel="Empirical cumulative distribution function"
    )


def compare_experiment_with_ecdf_square():
    name_files_reader = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    name_files_cam = ["Cal3105run0", "Cal3105run1", "Cal3105run2"]

    x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")

    ax = plt.axes(title="ECDF per different Run")
    for name_file_reader, name_file_cam in zip(name_files_reader, name_files_cam):
        x_test, y_test = dataset_generator.generate_dataset([name_file_reader], [name_file_cam],
                                                            dataset_generator.generate_dataset_base)

        ecdf_df = get_ecdf_dataset_squares(x_train, x_test, y_train, y_test,
                                           KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
                                           f'kNN_{name_file_reader}')
        index = ecdf_df.index.tolist()
        ax.step(np.array(index), ecdf_df[f'kNN_{name_file_reader}'], label=name_file_reader, where="post")

    ax.set_xlabel("squares")
    ax.set_ylabel("Empirical cumulative distribution function")
    plt.legend(loc='lower right')
    plt.show()


def compare_regressor_with_ecdf(train_dataset, test_dataset, name_file_reader, regressors=None, what_type_of_ecdf=0):
    """
    Compare the performance of different regressor with a run by ecdf
    :param name_file_reader: experiment where are  taken the data
    :param train_dataset: dataset where to train the regressors
    :param test_dataset: dataset to be used as testing
    :param regressors: regressors to compare
    :param what_type_of_ecdf:
        0: euclidean
        1: square
    :return: void
    """

    if regressors is None:
        regressors = CLASSIFIERS_DICT

    name_plot = ' '.join(name_file_reader)
    if what_type_of_ecdf == 0:
        ecdf_total = compare_regressor_with_ecdf_euclidean(train_dataset, test_dataset, regressors)

        ecdf_total.plot.line(
            title=f"ECDF {name_plot}",
            xlabel="(m)",
            ylabel="Empirical cumulative distribution function"
        )

        plt.savefig(f'plots/ecdf_euclidean_{name_plot}.png')
    if what_type_of_ecdf == 1:
        fig, ax = plt.subplots()
        ax.set_title(f"ECDF {name_plot}")
        ax = compare_regressor_with_ecdf_square(train_dataset, test_dataset, regressors, ax)

        plt.legend(loc='lower right')
        plt.show()

        plt.savefig(f'plots/ecdf_square_{name_plot}.png')


def compare_regressor_with_ecdf_euclidean(train_dataset, test_dataset, regressors):
    x_train = train_dataset[0]
    y_train = train_dataset[1]

    x_test = test_dataset[0]
    y_test = test_dataset[1]

    ecdf_total = pd.DataFrame()
    for regressor_name in regressors:
        ecdf_df = get_ecdf_regressor_dataset(x_train, x_test, y_train, y_test, regressors[regressor_name],
                                             regressor_name)
        ecdf_total = pd.concat([ecdf_total, ecdf_df], axis=1)

    ecdf_total = ecdf_total.interpolate(method='linear')

    return ecdf_total


def compare_regressor_with_ecdf_square(train_dataset, test_dataset, regressors, ax):
    x_train = train_dataset[0]
    y_train = train_dataset[1]

    x_test = test_dataset[0]
    y_test = test_dataset[1]

    for regressor_name in regressors:
        ecdf_df = get_ecdf_dataset_squares(x_train, x_test, y_train, y_test, regressors[regressor_name], regressor_name)

        index = ecdf_df.index.tolist()
        ax.step(np.array(index), ecdf_df[regressor_name], label=regressor_name, where="post")

    ax.set_xlabel("squares")
    ax.set_ylabel("Empirical cumulative distribution function")

    return ax


def compare_k_NNs(max_k, stride, train_dataset, test_dataset, name_file_readers, weights='distance',
                  what_type_of_ecdf=0):
    kNN_dict = regressors_lib.get_kNN_dict(max_k, stride, weights)

    compare_regressor_with_ecdf(train_dataset, test_dataset, name_file_readers, regressors=kNN_dict,
                                what_type_of_ecdf=what_type_of_ecdf)
