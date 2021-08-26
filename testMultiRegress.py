import pandas as pd

import config
import numpy as np
import utility
import plot_utility
import matplotlib.pyplot as plt
import dataset_generator
from scipy.stats import iqr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


NAMES = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree"]
CLASSIFIERS = [
    RandomForestRegressor(),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    DecisionTreeRegressor(random_state=0)
]


def performance_dataset(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.20)

    lor = y_test[:, 0]
    lop = y_test[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    errors = {}
    for name, clf in zip(NAMES, CLASSIFIERS):
        clf.fit(x_train, y_train)
        Z = clf.predict(x_test)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        errors[f'{name}--IQR-X'] = iqr(error_x)
        errors[f'{name}--IQR-Y'] = iqr(error_y)
        errors[f'{name}--MEDIAN X'] = np.median(error_x)
        errors[f'{name}--MEDIAN Y'] = np.median(error_y)

    return errors


def get_number_of_good_point(x_train, x_test, y_train, y_test):
    lor = y_test[:, 0]
    lop = y_test[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    good_points = {}
    for name, clf in zip(NAMES, CLASSIFIERS):
        clf.fit(x_train, y_train)
        Z = clf.predict(x_test)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

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
    lor = y_test[:, 0]
    lop = y_test[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    error_x_df = pd.DataFrame()
    error_y_df = pd.DataFrame()
    for name, clf in zip(NAMES, CLASSIFIERS):
        print("fit " + name)
        clf.fit(x_train, y_train)
        Z = clf.predict(x_test)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        error_x_df.insert(len(error_x_df.columns), name, error_x)
        error_y_df.insert(len(error_y_df.columns), name, error_y)

        df = pd.DataFrame({
            'diff x': error_x,
            'diff y': error_y
        })

        # df.plot.scatter(title=name, x='diff x', y='diff y')
        # df.plot.hexbin(title=f'{name} {title_add}', x='diff x', y='diff y', gridsize=20)
        df.plot.kde(title=f'{name} {title_add}')

    error_x_df.plot.box(title=f"Error x {title_add}")
    error_y_df.plot.box(title=f"Error y {title_add}")


if __name__ == "__main__":
    # X, y = utility.load_dataset_arff("datasetTrain0")
    # X, y = dataset_generator.generate_dataset_base_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test)
    #
    # X, y = dataset_generator.generate_dataset_with_mean_and_std_all()
    # x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.70)
    # plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="mean_std")

    x_train, y_train = dataset_generator.generate_dataset_base("BLE2605r", "2605r0")

    X_a = []
    y_a = []
    x, y = dataset_generator.generate_dataset_base("dati3105run0r", "Cal3105run0")
    X_a.append(x)
    y_a.append(y)
    x, y = dataset_generator.generate_dataset_base("dati3105run1r", "Cal3105run1")
    X_a.append(x)
    y_a.append(y)
    x, y = dataset_generator.generate_dataset_base("dati3105run2r", "Cal3105run2")
    X_a.append(x)
    y_a.append(y)

    x_test, y_test = dataset_generator.concatenate_dataset(X_a, y_a)

    plot_test_multi_regress(x_train, x_test, y_train, y_test)

    x_train, y_train = dataset_generator.generate_dataset_with_mean_and_std("BLE2605r", "2605r0")

    X_a = []
    y_a = []
    x, y = dataset_generator.generate_dataset_with_mean_and_std("dati3105run0r", "Cal3105run0")
    X_a.append(x)
    y_a.append(y)
    x, y = dataset_generator.generate_dataset_with_mean_and_std("dati3105run1r", "Cal3105run1")
    X_a.append(x)
    y_a.append(y)
    x, y = dataset_generator.generate_dataset_with_mean_and_std("dati3105run2r", "Cal3105run2")
    X_a.append(x)
    y_a.append(y)

    x_test, y_test = dataset_generator.concatenate_dataset(X_a, y_a)

    plot_test_multi_regress(x_train, x_test, y_train, y_test, title_add="mean_std")
