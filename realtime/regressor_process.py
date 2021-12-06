"""
Script that contains the body of the regressor process in the real time experiment

Different from the cnn and rnn, the regressor will be also trained.
To do that generate an arff dataset, that can be done with the last code in the dataset generator, and change it in line
52 and 56

Each time the last value filtered with kalman is used to get the inference.
"""
import time

import arff
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

import config
import utility
from moving_average import MovingAverageFilter

names = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree"]
regressions = [
    RandomForestRegressor(),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='uniform'),
    KNeighborsRegressor(config.N_NEIGHBOURS, weights='distance'),
    DecisionTreeRegressor(random_state=0)
]


def new_evaluete(v):
    ret = []
    array = np.array(v)
    array = array.reshape(1, -1)
    for clf in regressions:
        Z = clf.predict(array)
        tx, ty = utility.pol2cartS(Z[0][0], Z[0][1])
        ret.append([tx, ty])
        # ret.append([Z[:,0],Z[:,1]])
    return ret


def worker_evaluate_regressor(n, start_valuating, rssi_value, new_pos_regressor):
    print('Worker: ' + str(n))

    # datasetRY = arff.load(open('datasetTele1006ALLy0.arff'))

    # datasetRX = arff.load(open('datasetTele1006ALLx0.arff'))

    # datasetRY = arff.load(open('datasets/arff/datasetAllSquarey0.arff'))
    datasetRY = arff.load(open('datasets/arff/datasetTrainy0.arff'))
    dataRY = np.array(datasetRY['data'])

    # datasetRX = arff.load(open('datasets/arff/datasetAllSquarex0.arff'))
    datasetRX = arff.load(open('datasets/arff/datasetTrainx0.arff'))
    dataRX = np.array(datasetRX['data'])

    X = dataRX[:, :5]
    ax = np.array(dataRX[:, 5:6])
    ay = np.array(dataRY[:, 5:6])
    ry, py = utility.cart2pol(ax, ay)

    y = np.column_stack([ry, py])

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.80)

    for name, clf in zip(names, regressions):
        clf.fit(x_train, y_train)
        print("fit " + name)

    filtro_output_x = MovingAverageFilter(2)
    filtro_output_y = MovingAverageFilter(2)

    while True:
        time.sleep(1)
        if not start_valuating.value:
            continue

        new_pos = new_evaluete(rssi_value)

        b = 0
        totx = 0
        toty = 0
        for _ in names:
            x = new_pos[b][0]
            y = new_pos[b][1]
            new_pos_regressor[b] = [x, y]
            totx = totx + x
            toty = toty + y

            b += 1

        xc = totx / len(names)
        yc = toty / len(names)
        xc = xc.round(2)
        yc = yc.round(2)
        filtro_output_x.step(xc)
        filtro_output_y.step(yc)

        new_pos_regressor[b] = [filtro_output_x.current_state(),  filtro_output_y.current_state()]
