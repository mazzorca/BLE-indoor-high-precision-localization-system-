import socket
import multiprocessing
from multiprocessing import Process, Manager
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
import time
import arff
import torchvision
from PIL import Image
import torch
import itertools

import RSSI_image_converter
import config
import data_converter
from Configuration import cnn_config, dataset_config

from get_from_repeated_tune_search import get_params
from kalman import SingleStateKalmanFilter
from moving_average import MovingAverageFilter
import matplotlib.animation

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from cnns_models import ble_cnn
from cnn_testing import load_model as load_model_cnn
from rnns_testing import load_model as load_model_rnn
from rnns_models import ble


def cart2pol(x, y):
    rho = np.zeros(len(x))
    phi = np.zeros(len(y))
    for i in range(len(x)):
        rho[i] = np.sqrt(x[i] ** 2 + y[i] ** 2)
        phi[i] = np.arctan2(y[i], x[i])
    return (rho, phi)


def pol2cart(rho, phi):
    x = np.zeros(len(rho))
    y = np.zeros(len(phi))
    for i in range(len(x)):
        x[i] = rho[i] * np.cos(phi[i])
        y[i] = rho[i] * np.sin(phi[i])
    return (x, y)


def pol2cartS(rho, phi):
    x = 0
    y = 0
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def objective(x, a, b, c):
    return (a * x ** 2) + (b * x) + c


def new_evaluete1(v):
    ret = []
    array = np.array(v)
    array = array.reshape(1, -1)
    for clf in regressions:
        Z = clf.predict(array)
        tx, ty = pol2cart(Z[:, 0], Z[:, 1])
        tx = tx * 100
        ty = ty * 100
        ret.append([tx, ty])
        # ret.append([Z[:,0],Z[:,1]])
    return ret


def new_evaluete(v):
    ret = []
    array = np.array(v)
    array = array.reshape(1, -1)
    for clf in regressions:
        Z = clf.predict(array)
        tx, ty = pol2cartS(Z[0][0], Z[0][1])
        ret.append([tx, ty])
        # ret.append([Z[:,0],Z[:,1]])
    return ret


def evaluate_cnn(model_cnn, image_np, transform):
    img = Image.fromarray(image_np, 'L')
    img = img.convert('RGB')

    tensor_img = transform(img)
    tensor_img = tensor_img.view([1, 1, 24, 24])

    with torch.no_grad():
        pred = model_cnn(tensor_img)
        probability = torch.nn.functional.softmax(pred, dim=1)

        probability_np = probability.cpu().numpy()[0]
        indexs = probability_np.argsort()[-cnn_config.NUMBER_ARGMAX_EUCLIDEAN:]

        normalized_sum = np.sum(probability_np[indexs])

        x = 0
        y = 0
        for index in indexs:
            normalized_probability = probability_np[index] / normalized_sum

            contribution_x = config.SQUARES[index].centroid.x * normalized_probability
            contribution_y = config.SQUARES[index].centroid.y * normalized_probability
            x += contribution_x
            y += contribution_y

        ret = [x, y]

    return ret


def evaluate_rnn(model_rnn, matrix_np, transform):
    with torch.no_grad():
        tensor_matrix = transform(matrix_np)
        tensor_matrix = tensor_matrix.float()
        pred = model_rnn(tensor_matrix)

        np_pred = pred.view(2).numpy().reshape(1, 2)

        x = np_pred[0, 0]
        y = np_pred[0, 1]

        if x < 0:
            x = 0

        if x > 1.8:
            x = 1.8

        if y < 0:
            y = 0

        if y > 0.9:
            y = 0.9

        ret = [x, y]

    return ret


def new_client(clientsocket, addr):
    index = 0
    if addr[0] == ipclient[0]:
        index = 0
    if addr[0] == ipclient[1]:
        index = 1
    if addr[0] == ipclient[2]:
        index = 2
    if addr[0] == ipclient[3]:
        index = 3
    if addr[0] == ipclient[4]:
        index = 4

    p = Process(target=worker, args=(index, clientsocket, dataReader, packet,))
    p.start()


def worker(n, socket, k, p, ):
    print('Worker: ' + str(n))
    while True:
        msg = socket.recv(1024)
        if msg != "":
            x = msg.decode().split(";")
            if len(x) == 6:
                v = float(x[3])
                k[n].append(v)
                p[n] += 1
        else:
            break
    socket.close()


if __name__ == '__main__':

    multiprocessing.freeze_support()

    manager = Manager()
    s = socket.socket()
    port = 5000

    ipclient = ["192.168.1.2", "192.168.1.29", "192.168.1.48", "192.168.1.6", "192.168.1.26"]
    host = "192.168.1.19"
    connected = 0
    threads = []

    s.bind((host, port))  # Bind to the port
    s.listen(5)  # Now wait for client connection.
    print('Server started!')

    values = [0.0, 0.0, 0.0, 0.0, 0.0]

    packet = manager.list()
    for i in range(5):
        packet.append(0)

    dataReader = manager.list(
        [manager.list([-45]), manager.list([-45]), manager.list([-45]), manager.list([-45]), manager.list([-45])])

    # datasetRY = arff.load(open('datasetTele1006ALLy0.arff'))

    # datasetRX = arff.load(open('datasetTele1006ALLx0.arff'))

    datasetRY = arff.load(open('datasets/arff/datasetTrain0y0.arff'))
    dataRY = np.array(datasetRY['data'])

    datasetRX = arff.load(open('datasets/arff/datasetTrain0x0.arff'))
    dataRX = np.array(datasetRX['data'])

    X = dataRX[:, :5]
    ax = np.array(dataRX[:, 5:6])
    ay = np.array(dataRY[:, 5:6])
    ry, py = cart2pol(ax, ay)

    y = np.column_stack([ry, py])

    x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.80)
    n_neighbors = 100
    names = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree"]

    regressions = [
        RandomForestRegressor(),
        KNeighborsRegressor(n_neighbors, weights='uniform'),
        KNeighborsRegressor(n_neighbors, weights='distance'),
        DecisionTreeRegressor(random_state=0)
    ]

    for name, clf in zip(names, regressions):
        clf.fit(x_train, y_train)
        print("fit " + name)

    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.0001  # Process covariance 0.00001
    R = 0.5  # Measurement covariance 0.01
    x = -35  # Initial estimate
    P = 1  # Initial covariance 
    '''
    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.00001  # Process covariance 0.00001
    R = 0.01# Measurement covariance 0.01
    x = -45  # Initial estimate
    P = 1  # Initial covariance
    '''

    kalman_filters = []
    for i in range(5):
        k = SingleStateKalmanFilter(A, B, C, x, P, Q, R)
        kalman_filters.append(k)

    filtro_output_x = MovingAverageFilter(2)
    filtro_output_y = MovingAverageFilter(2)

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # initialize cnn # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    model_name = "ble"
    kalman_cnn = "kalman"
    transform_cnn = cnn_config.MODELS[model_name]["transform"]
    model_cnn = cnn_config.MODELS[model_name]["model"]

    params_cnn = {
        "wxh-stride": "20x20-10",
        "epoch": 20,
        "batch_size": 32,
        "lr": 0.01
    }

    model_name = f"{model_name}_{kalman_cnn}"
    parameters_saved = f"{model_name}/{int(params_cnn['epoch'])}-{params_cnn['lr']}-{int(params_cnn['batch_size'])}-{params_cnn['wxh-stride']}"
    model_cnn = load_model_cnn(model_cnn, parameters_saved)
    model_cnn.eval()

    image_np = np.zeros((20, 20))

    # kalman_filter_par = config.KALMAN_BASE
    # kalman_filter_cnn = data_converter.create_kalman_filter(kalman_filter_par)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # initialize rnn # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    kalman_rnn = "kalman"
    transform_rnn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    params_rnn = {
        "lr": 0.01,
        "lstm_size": 32,
        "linear_mul": 4
    }

    df_params, best_seed = get_params(f"{kalman_rnn}/rnn", list(params_rnn.keys()))
    for param in params_rnn.keys():
        params_rnn[param] = df_params.iloc[0][param]

    model_rnn = ble.BLErnn(int(params_rnn["linear_mul"]), int(params_rnn["lstm_size"]))
    model_rnn = load_model_rnn(model_rnn, kalman_rnn)
    model_rnn.eval()
    model_rnn = model_rnn.float()

    matrix_np = np.zeros((10, 5))
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    fig, ax = plt.subplots()
    x_plot, y_plot = [0], [0]
    sc = ax.scatter(x_plot, y_plot)
    plt.xlim(0, 180)
    plt.ylim(0, 90)
    # Major ticks every 20, minor ticks every 5
    major_ticks_x = np.arange(0, 181, 30)
    minor_ticks_x = np.arange(0, 181, 15)
    major_ticks_y = np.arange(0, 91, 30)
    minor_ticks_y = np.arange(0, 91, 15)

    ax.set_xticks(major_ticks_x)
    # ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    # ax.set_yticks(minor_ticks_y, minor=True)
    # And a corresponding grid
    ax.grid(which='both')
    plt.title("Tracciamento Real time Tag")
    plt.ylabel('Posizione Y')
    plt.xlabel('Posizione X')
    plt.legend('punto nello spazio')


    def animate(xt, yt):
        x_plot.append(xt)
        y_plot.append(yt)
        if len(x_plot) > 5:
            x_plot.pop(0)
            y_plot.pop(0)
        sc.set_offsets(np.c_[x_plot, y_plot])
        plt.show(block=False)
        plt.pause(0.001)

    print('Waiting for clients...')

    x = 0
    y = 0
    h = 4
    w = 20
    index_rnn = 0
    max_index_rnn = 10

    first_values = True
    c = []
    while True:
        try:
            if connected < 5:
                print("aspetto connessioni")
                new_c, addr = s.accept()
                c.append(new_c)
                print('Got connection from', addr)
                connected += 1
                new_client(new_c, addr)
            else:

                time.sleep(1)
                p = []
                for i in range(5):
                    for d in dataReader[i]:
                        kalman_filters[i].step(0, dataReader[i].pop(0))
                    values[i] = kalman_filters[i].current_state()
                    p.append(packet[i])
                    packet[i] = 0

                new_pos = new_evaluete(values)

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # evaluate cnn # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                min = dataset_config.NORM_MIN_K
                max = dataset_config.NORM_MAX_K

                dividend = max - min

                cnn_rssi_values = []
                for RSSI_value in values:
                    cnn_rssi_values.append(((RSSI_value - min) / dividend)*255)

                print("RSSI_value: ", cnn_rssi_values)

                # for reader_num in range(5):
                #     image_np = np.concatenate([image_np.flatten('F')[3:], [4, 4, 4]]).reshape(3, 3, order='F')
                #     image_np[x + reader_num, y] = values[reader_num]

                if first_values:
                    init_rssi_values = list(itertools.chain(cnn_rssi_values, cnn_rssi_values, cnn_rssi_values, cnn_rssi_values))
                    image_np = np.full((h*5, w), init_rssi_values, dtype=np.uint8).transpose()
                    if config.debug_cnn:
                        img = Image.fromarray(image_np, 'L')
                        img = img.convert('RGB')
                        img.show()

                new_values_rssi = np.array(cnn_rssi_values, dtype=np.uint8)
                image_np = np.concatenate([image_np.flatten('F')[5:], new_values_rssi]).reshape(h*5, w, order='F')

                if config.debug_cnn:
                    img = Image.fromarray(image_np, 'L')
                    img = img.convert('RGB')
                    img.show()

                # x += 1
                #
                # if x == h:
                #     y += 1
                #     x = 0
                #
                # if y == w:
                #     y = 0

                new_pos_cnn = evaluate_cnn(model_cnn, image_np, transform_cnn)
                print(f"cnn {model_name}: x", new_pos_cnn[0], "y", new_pos_cnn[1])
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # evaluate rnn # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
                # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

                rnn_rssi_values = []
                for RSSI_value in values:
                    rnn_rssi_values.append((RSSI_value + abs(min)))

                # for reader_num in range(5):
                #     matrix_np[index_rnn, reader_num] = values[reader_num]

                # index_rnn += 1
                # if index_rnn == 10:
                #     index_rnn = 0

                if first_values:
                    matrix_np = np.full((10, 5), rnn_rssi_values)
                    if config.debug_rnn:
                        matrix_np_d = np.full((10, 5), rnn_rssi_values, dtype=np.uint8)
                        img_d_rnn = Image.fromarray(matrix_np_d, 'L')
                        img_d_rnn = img_d_rnn.convert('RGB')
                        img_d_rnn.show()

                matrix_np = np.concatenate([matrix_np.flatten()[5:], rnn_rssi_values]).reshape(10, 5)

                if config.debug_rnn:
                    img_d_rnn = Image.fromarray(matrix_np, 'L')
                    img_d_rnn = img_d_rnn.convert('RGB')
                    img_d_rnn.show()

                new_pos_rnn = evaluate_rnn(model_rnn, matrix_np, transform_rnn)
                print(f"rnn {kalman_rnn}: x", new_pos_rnn[0], "y", new_pos_rnn[1])
                # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

                b = 0
                totx = 0
                toty = 0
                for name in names:
                    xctemp = new_pos[b][0]
                    yctemp = new_pos[b][1]
                    print(str(name) + " x:" + str(xctemp.round(2)) + " y: " + str(yctemp.round(2)), flush=True)
                    totx = totx + new_pos[b][0]
                    toty = toty + new_pos[b][1]
                    b += 1
                xc = totx / len(names)
                yc = toty / len(names)
                xc = xc.round(2)
                yc = yc.round(2)
                filtro_output_x.step(xc)
                filtro_output_y.step(yc)
                # print("K Neighbors Regressor Media" +" x:"+str()+" y: "+str(toty/len(names)), flush=True)
                print("Posizione" + " x:" + str(filtro_output_x.current_state()) + " y: " + str(
                    filtro_output_y.current_state()), flush=True)
                animate(filtro_output_x.current_state(), filtro_output_y.current_state())
                # print("Posizione" +" x:"+str(xc)+" y: "+str(yc), flush=True)
                # animate(xc,yc)
                print("raccolti : " + str(p[0]) + " " + str(p[1]) + " " + str(p[2]) + " " + str(p[3]) + " " + str(p[4]),
                      flush=True)

                for i in range(5):
                    values[i] = kalman_filters[i].current_state()
                print("RSSI : " + str(values[0]) + " " + str(values[1]) + " " + str(values[2]) + " " + str(
                    values[3]) + " " + str(values[4]), flush=True)

                print("\n")

                first_values = False
        except KeyboardInterrupt:
            for connection in c:
                if connection:
                    connection.close()
            break
