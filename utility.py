from shapely.geometry import Point

import os
import errno
import csv

import config
import data_extractor
import numpy as np
from pylab import ylim, title, ylabel, xlabel
import matplotlib.pyplot as plt

from data_converter import fixReader, cutReader, create_kalman_filter
import pandas as pd
from scipy.io import arff
import random
import arff

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


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


def get_euclidean_distance(a_x, a_y):
    lox = a_x[:, 0]
    loy = a_x[:, 1]
    lpx = a_y[:, 0]
    lpy = a_y[:, 1]

    error_x = abs(np.subtract(lpx, lox))
    error_y = abs(np.subtract(lpy, loy))

    error_x = np.power(error_x, 2)
    error_y = np.power(error_y, 2)
    errors = np.add(error_x, error_y)
    errors = np.sqrt(errors)

    return errors


def equalize_data_with_nan(data):
    max_dim = max([len(elem) for elem in data])
    print(max_dim)

    for vec in data:
        vec.extend([np.nan] * (max_dim - len(vec)))

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


def get_means_and_bounds(chunks):
    raw_means = []
    bounds_up = []
    bounds_down = []

    for chunks_reader in chunks:
        raw_mean = []
        for raw_chunk in chunks_reader:
            raw_mean.append(np.mean(raw_chunk['RSSI Value']))
        raw_means.append(raw_mean)

    for reader in raw_means:
        bounds_up.append([elem + config.OUTLIERS_BAND for elem in reader])
        bounds_down.append([elem - config.OUTLIERS_BAND for elem in reader])

    return raw_means, bounds_up, bounds_down


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


def remove_outliers(kalman_chunks, bounds_ups, bounds_downs):
    data_without_outliers = []
    for i, chunks in enumerate(kalman_chunks):
        data_without_outliers.append([])
        for j, chunk in enumerate(chunks):
            df = pd.DataFrame(chunk)
            data_without_outliers[i].extend(df.clip(bounds_ups[i][j], bounds_downs[i][j])['RSSI Value'].to_list())

    return data_without_outliers


def add_mean_and_std(kalman_data, raw_chunks):
    means = []
    stds = []
    for reader_number, raw_chunk_reader in enumerate(raw_chunks):
        means.append([])
        stds.append([])
        for chunk in raw_chunk_reader:
            df_temp = chunk.copy()
            chunk['mean'] = df_temp.rolling(int((len(chunk) / 10))).std()
            means[reader_number].extend(chunk['mean'].tolist())
            chunk['std'] = df_temp.rolling(int((len(chunk) / 10))).mean()
            stds[reader_number].extend(chunk['std'].tolist())

    kalman_data.extend(means)
    kalman_data.extend(stds)

    return kalman_data


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
    filep = "datasets/arff/dataset" + name + "p.arff"
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

    filex = "datasets/arff/dataset" + name + "x0.arff"
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

    filey = "datasets/arff/dataset" + name + "y0.arff"
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
    datiReaderCut, datiTeleCut, _ = cutReader(datiReaderNew, datiTele, indtaglio)

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


def get_square_number_array(lx, ly):
    squares = config.SQUARES

    squares_x = []
    squares_y = []
    for x, y in zip(lx, ly):
        square_x, square_y = get_square_number(x, y, squares)

        squares_x.append(square_x)
        squares_y.append(square_y)

    return squares_x, squares_y


def get_squarex_and_squarey(i):
    square_x = i % 6
    square_y = int(i / 6)

    return square_x, square_y


def get_square_number(x, y, squares):
    point = Point(x, y)

    i = 0
    find = False
    for i in range(len(squares)):
        if squares[i].contains(point):
            find = True
            break

    if not find:
        index = 0
        dist_min = 200
        for i in range(len(squares)):
            p_predicted = np.array([x, y])
            p_square = np.array([squares[i].centroid.x, squares[i].centroid.y])
            dist = np.linalg.norm(p_predicted - p_square)

            if dist < dist_min:
                dist_min = dist
                index = i

        i = index

    square_x, square_y = get_squarex_and_squarey(i)

    return square_x, square_y


def check_and_if_not_exists_create_folder(filename):
    if not os.path.exists(os.path.dirname(filename)):
        try:
            os.makedirs(os.path.dirname(filename))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise


def append_to_csv(file_name_csv, row_to_append):
    with open(file_name_csv, 'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(row_to_append)
