import pandas as pd

import config
import numpy as np
import utility
import plot_utility
import matplotlib.pyplot as plt
from scipy.stats import iqr

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor


NAMES = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree", "MLP"]
CLASSIFIERS = [
    RandomForestRegressor(),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    DecisionTreeRegressor(random_state=0),
    MLPRegressor(random_state=1, max_iter=50000)
]


def performance_dataset(X, y):
    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.50)

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


if __name__ == "__main__":
    X, y = utility.load_dataset_arff("datasetTrain0")

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.50)

    lor = y[:, 0]
    lop = y[:, 1]
    lox, loy = utility.pol2cart(lor, lop)

    error_x_df = pd.DataFrame()
    error_y_df = pd.DataFrame()
    for name, clf in zip(NAMES, CLASSIFIERS):
        print("fit " + name)
        clf.fit(x_train, y_train)
        Z = clf.predict(X)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        error_x_df.insert(len(error_x_df.columns), name, error_x)
        error_y_df.insert(len(error_y_df.columns), name, error_y)

        df = pd.DataFrame({
            'predicted x': lpx,
            'optimal x': lox,
            'predicted y': lpy,
            'optimal y': loy
        })

        fig, axes = plt.subplots(2, 1)
        fig.suptitle(name)
        df.iloc[:, :2].plot(ax=axes[0])
        df.iloc[:, 2:].plot(ax=axes[1])

    error_x_df.plot.box(title="Error x")
    error_y_df.plot.box(title="Error y")
