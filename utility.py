import config
import data_extractor
import numpy as np
from pylab import ylim, title, ylabel, xlabel
import matplotlib.pyplot as plt
from kalman import SingleStateKalmanFilter
from moving_average import MovingAverageFilter
import csv
import pandas as pd
from scipy.io import arff
import sys
import random
import math
import time
import arff
from filterpy.kalman import KalmanFilter
from moving_average import MovingAverageFilter
import matplotlib.animation

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf
import os

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
import tensorflow.keras
from scipy.optimize import curve_fit
from tensorflow.keras.models import load_model
from keras import layers
from keras import Input
from keras.models import Model


# from keras.utils import to_categorical


# funzione per la conversione da coordinate cartesiane a polari
def cart2pol(x, y):
    rho = np.zeros(len(x))
    phi = np.zeros(len(y))
    for i in range(len(x)):
        rho[i] = np.sqrt(x[i] ** 2 + y[i] ** 2)
        phi[i] = np.arctan2(y[i], x[i])
    return rho, phi


# funzione per la conversione da coordinate polari a cartesiane
def pol2cart(rho, phi):
    x = np.zeros(len(rho))
    y = np.zeros(len(phi))
    for i in range(len(x)):
        x[i] = rho[i] * np.cos(phi[i])
        y[i] = rho[i] * np.sin(phi[i])
    return x, y


# funzione per la conversione da coordinate polari a cartesiane (singolarmenti)
def pol2cartS(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return x, y


def equalize_data_with_nan(data):
    max_dim = max([len(elem) for elem in data])
    print(max_dim)

    for vec in data:
        vec.extend([np.nan]*(max_dim - len(vec)))

    return data


def create_or_insert_in_list(building_dict, key, value):
    """
    Create the list or append a value on it
    :param building_dict: building dict
    :param key: key to save value
    :param value: thee value to add to the list
    :return: void
    """
    if key not in building_dict:
        building_dict[key] = list()
    building_dict[key].append(value)


def moving_average(x, w):
    return np.convolve(x, np.ones(w), 'valid') / w


def change_Q_R(kalman_filter_par, Q, R):
    kalman_filter_par['R'] = Q
    kalman_filter_par['Q'] = R

    return kalman_filter_par

# funzione per fare la predizione dei valori V con l'estimantore est
def new_evaluete(v, est):
    ret = []
    array = np.array(v)
    array = array.reshape(1, -1)
    Z = est.predict(array)
    # print(Z)
    tempx, tempy = (pol2cart(Z[:, 0], Z[:, 1]))
    # tempx,tempy =(Z[0],Z[1])

    # mx = mx/100 #convert cm--> metri
    # my = my/100 #convert cm--> metri
    return tempx, tempy


def new_evaluete_P(v, est):
    ret = []
    Z = est.predict(v)
    print(Z)
    return Z


# funzione per fare la predizione dei vettore V con il regressore regress
def new_evaluete_regress(v, regress):
    mx = 0
    my = 0
    array = np.array(v)
    array = array.reshape(1, -1)
    for clf in regress:
        Z = clf.predict(array)
        # print(Z)
        tempx, tempy = pol2cartS(Z[0][0], Z[0][1])
        mx = mx + tempx
        my = my + tempy
    mx = mx / len(regress)
    my = my / len(regress)

    return mx, my


# funzione per fare la creazione del regressore
def createRegress(name):
    filex = "datasetTele" + name + "x0.arff"
    datasetRX = arff.load(open(filex))
    dataRX = np.array(datasetRX['data'])

    filey = "datasetTele" + name + "y0.arff"
    datasetRY = arff.load(open(filey))
    dataRY = np.array(datasetRY['data'])

    X = dataRX[:, :5]
    ax = np.array(dataRX[:, 5:6])
    ay = np.array(dataRY[:, 5:6])
    ry, py = cart2pol(ax, ay)
    y = np.column_stack([ry, py])

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.9)

    n_neighbors = 20
    offset = 10
    names = [
        "K Neighbors Regressor 50",
        "K Neighbors Regressor 100",
        "K Neighbors Regressor 150",
        "K Neighbors Regressor 200",
        "K Neighbors Regressor 250",
        "K Neighbors Regressor 300",
    ]

    regressions = [
        KNeighborsRegressor(n_neighbors, weights='distance'),
        KNeighborsRegressor(n_neighbors + offset * 2, weights='distance'),
        KNeighborsRegressor(n_neighbors + offset * 3, weights='distance'),
        KNeighborsRegressor(n_neighbors + offset * 4, weights='distance'),
        KNeighborsRegressor(n_neighbors + offset * 5, weights='distance'),
        KNeighborsRegressor(n_neighbors + offset * 6, weights='distance'),
    ]

    for name, clf in zip(names, regressions):
        clf.fit(x_train, y_train)
        print("fit " + name)
    return regressions


# funzioni per plottare i dei diversi dati
def printPuliti(datidaplottare):
    for i in range(5):
        title(" r " + str(i))
        ylabel('rssi')
        xlabel('Sample')
        ylim([-50, 20])
        for t in range(18):
            r = random.random()
            b = random.random()
            g = random.random()
            color = (r, g, b)
            plt.plot(datidaplottare[i][t], c=color, linewidth=2.0, label="p" + str(t))
        plt.legend()
        # Show the plot 
        plt.show()


def printPlot(d1):
    plt.plot(len(d1), 'r*')
    title(" dati ")
    ylabel('valore')
    xlabel('Sample')
    plt.plot(d1, 'b', linewidth=2.0, label="d")
    plt.legend()
    plt.show()


def printReader(d1, d2):
    for i in range(5):
        plt.plot(len(d1[i]), 'r*')
        title(" Reader" + str(i + 1))
        ylabel('rssi')
        xlabel('Sample')
        ylim([-60, -10])
        plt.plot(d1[i], 'r', linewidth=2.0, label="k")
        plt.plot(d2[i], 'b', linewidth=2.0, label="d")
        plt.legend()
        # Show the plot
        plt.show()


def printDatiTaglio(dati, confronto, taglio):
    dim = len(dati)
    (fig, axs) = plt.subplots(2, 1)

    for i in range(dim):
        axs[1].plot(len(dati[i]), '*r')
        title(" dati" + str(i + 1))
        ylabel('value')
        ylim([-50, -20])
        xlabel('Sample')
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        axs[1].plot(dati[i], c=color, linewidth=2.0, label="reader" + str(i))

    axs[0].plot(confronto[2], 'b', linewidth=2.0, label="confronto x")
    axs[0].plot(confronto[3], 'r', linewidth=2.0, label="confronto y")
    axs[0].plot(confronto[4], 'r', linewidth=2.0, label="confronto z")
    temp = [element * (-1) for element in taglio]
    axs[0].plot(temp, 'g', linewidth=2.0, label="confronto z")

    axs[1].plot(taglio, 'r', linewidth=2.0, label=" z ")

    plt.show()


def printDati(dati, confronto):
    dim = len(dati)
    (fig, axs) = plt.subplots(2, 1)

    for i in range(dim):
        axs[1].plot(len(dati[i]), '*r')
        title(" dati" + str(i + 1))
        ylabel('value')
        ylim([-50, -20])
        xlabel('Sample')
        r = random.random()
        b = random.random()
        g = random.random()
        color = (r, g, b)
        axs[1].plot(dati[i], c=color, linewidth=2.0, label="reader" + str(i))

    axs[0].plot(confronto[0], 'b', linewidth=2.0, label="confronto x")
    axs[0].plot(confronto[1], 'r', linewidth=2.0, label="confronto y")
    plt.legend()
    plt.show()


def printDatiConf(dati1, dati2):
    dim = len(dati2)
    (fig, axs) = plt.subplots(1, dim)

    for i in range(dim):
        axs[i].plot(dati1[i], 'b', linewidth=2.0, label="confronto 1 " + str(i))
        axs[i].plot(dati2[i], 'r', linewidth=2.0, label="confronto 2 " + str(i))

    plt.legend()
    plt.show()


def printRealEvalXY(real, eval):
    (fig, axs) = plt.subplots(1, 2)

    axs[0].plot(real[0], 'b', linewidth=2.0, label="real x")
    axs[0].plot(eval[0], 'r', linewidth=2.0, label="eval x")
    axs[1].plot(real[1], 'b', linewidth=2.0, label="real y")
    axs[1].plot(eval[1], 'r', linewidth=2.0, label="eval y")
    plt.show()


def printError(errorx, errory):
    title(" errore ")
    ylabel('errore (cm)')
    xlabel('Sample')
    tempx = [element * 100 for element in errorx]
    tempy = [element * 100 for element in errory]
    plt.plot(tempx, 'r', linewidth=2.0, label="errore x")
    plt.plot(tempy, 'b', linewidth=2.0, label="errore y")
    plt.legend()
    plt.show()


# funzione per la conversione dei dati da file .emt in una lista di array ["Frame","Time","Mk0.X","Mk0.Y","Mk0.Z"]
# effettuo la pulizia dei valori NAN e taglio i valori con z > 0.05 metri quando passo da una postazione all'altra
def convertEMT(namefile):
    f = open(f"dati/{namefile}.emt")
    lines = f.readlines()
    dataSplit = []
    count = 0
    # label=["Frame","Time","Mk0.X","Mk0.Y","Mk0.Z","Mk1.X","Mk1.Y","Mk1.Z","Mk2.X","Mk2.Y","Mk2.Z"]
    label = ["Frame", "Time", "Mk0.X", "Mk0.Y", "Mk0.Z"]
    lab = ["X", "Y", "Z"]
    # Strips the newline character
    for line in lines:
        count += 1
        if count > 11:  # 10 con label 11 senza label
            l = line.strip()
            t = l.split()

            # Frame Time Mk0.X Mk0.Y Mk0.Z
            for i in range(len(t)):
                if count == 12:  # 11 con label 12 senza label
                    d = [float(t[i])]
                    dataSplit.append(d)
                else:
                    dataSplit[i].append(float(t[i]))

    dataSplitNew = []

    for g in range(len(dataSplit)):
        d = []
        dataSplitNew.append(d)

    res = np.isnan(dataSplit[2])  # controllo sulla X

    for j in range(len(res)):
        if not res[j]:
            if dataSplit[4][j] < 0.05:
                for i in range(len(dataSplit)):
                    dataSplitNew[i].append(dataSplit[i][j])

    return dataSplitNew


def create_kalman_filter(kalman_filter_par):
    kalman_filter = KalmanFilter(dim_x=1, dim_z=1)
    kalman_filter.F = np.array([[kalman_filter_par['A']]])
    kalman_filter.H = np.array([[kalman_filter_par['C']]])
    kalman_filter.R = np.array([[kalman_filter_par['R']]])
    kalman_filter.Q = np.array([[kalman_filter_par['Q']]])

    return kalman_filter


# funzione per la conversione dei dati da file .csv dai 5 reader in una lista di array [R1,R2,R3,R4,R5]
# effettuo il filtraggio dei dati con il filtro di kalman e restituisco i dati filtrati e il timestamp
def extract_and_apply_kalman_csv(namefile, kalman_filter_par=None):
    dataset = []
    dataset_time = []

    if kalman_filter_par is None:
        kalman_filter_par = config.KALMAN_BASE

    # create kalman filter
    kalman_filter = create_kalman_filter(kalman_filter_par)

    for i in range(5):
        # reset the filter
        kalman_filter.x = np.array([kalman_filter_par['x']])
        kalman_filter.P = np.array([[kalman_filter_par['P']]])

        raw_data, raw_time = data_extractor.get_raw_rssi_csv_reader(f"dati/{namefile}{str(i + 1)}.csv")

        dataset_time.append(raw_time)

        dataset.append([])
        for raw_point in raw_data:
            kalman_filter.predict()
            kalman_filter.update(raw_point)
            dataset[i].append(kalman_filter.x[0])

    return dataset, dataset_time


def apply_kalman_filter(no_kalman_data, kalman_filter_par):
    """
    Apply  the kalman filter on the raw data
    :param no_kalman_data:
    :type kalman_filter_par: dict
    """
    kalman_data = []

    kalman_filter = create_kalman_filter(kalman_filter_par)

    for i, raw_data_reader in enumerate(no_kalman_data):
        # reset the filter
        kalman_filter.x = np.array([kalman_filter_par['x']])
        kalman_filter.P = np.array([[kalman_filter_par['P']]])

        kalman_data.append([])
        for raw_value in raw_data_reader:
            kalman_filter.predict()
            kalman_filter.update(raw_value)
            kalman_data[i].append(kalman_filter.x[0])

    return kalman_data


def get_chunk(rssi_data, index_cut, chunk_num=-1):
    """
    Divide the rssi_data in  chunks
    :param rssi_data: data to split
    :param index_cut: index where to split
    :param chunk_num: a list of reader num list containing 3 options:
                -2, return a dataframe with all the rssi for all position
                -1, return a list of dataframe one for all position
                >0, return a dataframe for the position selected
    :return: a list of list
    """
    if chunk_num == -2:
        chunks = [pd.DataFrame({"RSSI Value": rssi}) for rssi in rssi_data]
        return chunks

    chunks = [[] for _ in range(5)]
    for reader_num, chunk in enumerate(chunks):
        reader_cut = index_cut[reader_num]

        for _, row in reader_cut.iterrows():
            cut_rssi = rssi_data[reader_num][row['start']:(row['end'] + 1)]
            chunk.append(pd.DataFrame({"RSSI Value": cut_rssi}))

    if chunk_num == -1:
        return chunks

    selected_chunk = [chunk[chunk_num] for chunk in chunks]
    return selected_chunk


def get_index_start_and_end_position(time):
    indextaglio_reader = []

    for reader_num in range(5):
        start_index = [0]
        end_index = []
        for i in range(len(time[reader_num]) - 1):
            if abs(time[reader_num][i] - time[reader_num][i + 1]) > 5:  # 5 secondi
                start_index.append(i + 1)
                end_index.append(i)
        end_index.append(len(time[reader_num]) - 1)

        df = pd.DataFrame({"start": start_index, "end": end_index})
        indextaglio_reader.append(df)

    return indextaglio_reader


def get_index_taglio_reader(time):
    indextaglio_reader = []

    for j in range(5):
        indextaglio_reader.append([0])
        for i in range(len(time[j]) - 1):
            if abs(time[j][i] - time[j][i + 1]) > 5:  # 5 secondi
                indextaglio_reader[j].append(i)
        indextaglio_reader[j].append(len(time[j]) - 1)

    return indextaglio_reader


# Funzione per il taglio dei dati provenienti dai 5 reader, andando ad allineare nel tempo i dati raccolti dalle
# telecamere e i dati raccolti dai reader nel tempo
def fixReader(dati, time, tele):
    newData = []
    newTime = []
    indextaglio = [100]
    for i in range(len(tele[2]) - 1):  # len(tele[2]) = lunghezza valori nella x
        if (abs(tele[2][i] - tele[2][i + 1]) > 0.1) | (abs(tele[3][i] - tele[3][i + 1]) > 0.1):
            indextaglio.append(i)

    indextaglio.append(len(tele[2]) - 1)

    indextaglio_reader = get_index_taglio_reader(time)

    for i in range(len(dati)):
        newData.append([])
        newTime.append([])
        for j in range(len(indextaglio_reader[i]) - 1):
            tot_data_inserted = 0
            dim_data_reader = indextaglio_reader[i][j + 1] - indextaglio_reader[i][j]  # dimensione dei dati raccolti
            # nel punto j-esimo dal BLE
            dim_data_tele = indextaglio[j + 1] - indextaglio[j]  # dimensione dei dati raccolti nel punto j-esimo
            # dalle tele
            fattore = math.trunc(dim_data_tele / dim_data_reader)

            for t in range(dim_data_reader):
                for x in range(fattore + 1):
                    if tot_data_inserted < dim_data_tele:
                        newData[i].append(dati[i][indextaglio_reader[i][j] + t])
                        newTime[i].append(time[i][indextaglio_reader[i][j] + t])
                        tot_data_inserted = tot_data_inserted + 1

    return newData, newTime, indextaglio


# funziona per il taglio dei dati per via del ritardo che introduce il filtro di kalman
def cutReader(dati, tele, ind):
    newData = [[], [], [], [], []]
    newTele = [[], []]
    new_data = [0, 0, 0, 0, 0]
    old_data = [0, 0, 0, 0, 0]

    for j in range(len(ind) - 1):
        dim = ind[j + 1] - ind[j]
        offset = math.trunc(dim / 3)
        # print(offset)
        lun = offset
        for t in range(lun):
            for i in range(len(dati)):
                if ind[j] + offset + t < len(dati[i]):
                    new_data[i] = dati[i][ind[j] + offset + t]

            if not np.array_equal(new_data, old_data):
                for i in range(len(dati)):
                    newData[i].append(dati[i][ind[j] + offset + t])
                for i in range(5):
                    old_data[i] = new_data[i]
                newTele[0].append(tele[2][ind[j] + offset + t])
                newTele[1].append(tele[3][ind[j] + offset + t])

    # for i in range (5):
    # print(len(newData[i]))
    return newData, newTele


def evalueteAll(dati, regress):
    v = [[], [], [], [], []]
    ret = [[], []]
    for i in range(len(dati[0])):
        for j in range(len(dati)):
            v[j] = dati[j][i]
        tx, ty = new_evaluete_regress(v, regress)
        ret[0].append(tx)
        ret[1].append(ty)
    return ret


def evalueteAllKeras(dati, regress):
    v = [[], [], [], [], []]
    ret = [[], []]
    for i in range(len(dati[0])):
        for j in range(len(dati)):
            v[j] = dati[j][i]
        tx, ty = new_evaluete(v, regress)
        ret[0].append(tx)
        ret[1].append(ty)
    return ret


def evalueteAllKerasSep(dati, model):
    v = [[], [], [], [], []]
    ret = [[], []]
    for i in range(len(dati[0])):
        for j in range(len(dati)):
            v[j] = dati[j][i]
        array = np.array(v)
        array = array.reshape(1, -1)
        Z = model.predict(array)
        ret.append(Z)
        for i in range(4):
            ret[i].append(Z[0][i])
    return ret


def calError(datiTeleCut, resEval):
    error = [[], []]
    numerrorx = 0
    numerrory = 0
    for i in range(len(datiTeleCut[0])):
        error[0].append(abs(datiTeleCut[0][i] - resEval[0][i]))
        error[1].append(abs(datiTeleCut[1][i] - resEval[1][i]))
        if error[0][i] > 0.05:
            numerrorx = numerrorx + 1
        if error[1][i] > 0.05:
            numerrory = numerrory + 1

    print("errore x " + str(numerrorx / len(resEval[0])))
    print("errore y " + str(numerrory / len(resEval[1])))
    printError(error[0], error[1])


def convert2Point(ottimo):
    pass


def saveDataArff(dati, ottimo, name):
    filep = "dataset" + name + "p.arff"
    with open(filep, "w") as fp:
        fp.write('''@RELATION point

    @ATTRIBUTE reader1	REAL
    @ATTRIBUTE reader2 	REAL
    @ATTRIBUTE reader3 	REAL
    @ATTRIBUTE reader4	REAL
    @ATTRIBUTE reader5	REAL
    @ATTRIBUTE class 	{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17}

    @DATA
    ''')

    filex = "dataset" + name + "x0.arff"
    with open(filex, "w") as fp:
        fp.write('''@RELATION point

    @ATTRIBUTE reader1	REAL
    @ATTRIBUTE reader2 	REAL
    @ATTRIBUTE reader3 	REAL
    @ATTRIBUTE reader4	REAL
    @ATTRIBUTE reader5	REAL
    @ATTRIBUTE class 	REAL

    @DATA
    ''')

    filey = "dataset" + name + "y0.arff"
    with open(filey, "w") as fp:
        fp.write('''@RELATION point

    @ATTRIBUTE reader1	REAL
    @ATTRIBUTE reader2 	REAL
    @ATTRIBUTE reader3 	REAL
    @ATTRIBUTE reader4	REAL
    @ATTRIBUTE reader5	REAL
    @ATTRIBUTE class 	REAL

    @DATA
    ''')

    for i in range(len(ottimo[0])):
        all_x = "{},{},{},{},{},{}\n".format(dati[0][i], dati[1][i], dati[2][i], dati[3][i], dati[4][i], ottimo[0][i])
        all_y = "{},{},{},{},{},{}\n".format(dati[0][i], dati[1][i], dati[2][i], dati[3][i], dati[4][i], ottimo[1][i])

        with open(filex, "a") as myfilex:
            myfilex.write(all_x)
        with open(filey, "a") as myfiley:
            myfiley.write(all_y)

    # punti = convert2Point(ottimo)
    # for i in range(len(punti)):
    #     all_p = "{},{},{},{},{},{}\n".format(dati[0][i], dati[1][i], dati[2][i], dati[3][i], dati[4][i], punti[i])
    #     with open(filep, "a") as myfilep:
    #         myfilep.write(all_p)


def takeData(nameCSV, nameEMT):
    datiReader, datiTimeReader = extract_and_apply_kalman_csv(nameCSV)

    datiTele = convertEMT(nameEMT)

    datiReaderNew, datiTimeReaderNew, indtaglio = fixReader(datiReader, datiTimeReader, datiTele)

    # printDati(datiReaderNew,(datiTele[2],datiTele[3]))
    datiReaderCut, datiTeleCut = cutReader(datiReaderNew, datiTele, indtaglio)

    return datiReaderCut, datiTeleCut


def load_dataset_arff(dataset_name):
    datasetRY = arff.load(open(f'{dataset_name}y0.arff'))
    dataRY = np.array(datasetRY['data'])

    datasetRX = arff.load(open(f'{dataset_name}x0.arff'))
    dataRX = np.array(datasetRX['data'])

    X = dataRX[:, :5]
    ax = np.array(dataRX[:, 5:6])
    ay = np.array(dataRY[:, 5:6])
    ry, py = cart2pol(ax, ay)

    y = np.column_stack([ry, py])

    return X, y
