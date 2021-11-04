import socket
import multiprocessing
from multiprocessing import Process, Manager, Value
import numpy as np
import matplotlib.pyplot as plt
import time
import arff
import json

import utility

from kalman import SingleStateKalmanFilter
from moving_average import MovingAverageFilter

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from realtime.cnn_process import worker_evaluate_cnn


def objective(x, a, b, c):
    return (a * x ** 2) + (b * x) + c


def new_evaluete1(v):
    ret = []
    array = np.array(v)
    array = array.reshape(1, -1)
    for clf in regressions:
        Z = clf.predict(array)
        tx, ty = utility.pol2cart(Z[:, 0], Z[:, 1])
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
        tx, ty = utility.pol2cartS(Z[0][0], Z[0][1])
        ret.append([tx, ty])
        # ret.append([Z[:,0],Z[:,1]])
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


def load_trajectory(trajectory_name):
    if not trajectory_name:
        return None

    with open(f'{trajectory_name}.json', ) as f:
        data = json.load(f)

    return data


def worker_evaluate_regressor(n, start_valuating, rssi_value, new_pos_regressor):
    print('Worker: ' + str(n))


if __name__ == '__main__':
    multiprocessing.freeze_support()
    plt.ion()

    manager = Manager()
    s = socket.socket()
    port = 5000

    start_valuating = Value('i', False)
    rssi_value_cnn = manager.list([
        [-45],
        [-45],
        [-45],
        [-45],
        [-45]
    ])
    new_pos_cnn = manager.list([0, 0])
    proc_cnn = Process(target=worker_evaluate_cnn, args=("cnn", start_valuating, rssi_value_cnn, new_pos_cnn,))
    proc_cnn.start()

    rssi_value_rnn = manager.list([-45, -45, -45, -45, -45])
    new_pos_rnn = manager.list([0, 0])
    # proc_rnn = Process(target=worker_evaluate_rnn, args=("rnn", start_valuating, rssi_value_rnn, new_pos_rnn,))
    # proc_rnn.start()

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

    dataReader = manager.list([
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45])
    ])

    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.00001  # Process covariance 0.00001
    R = 0.01  # Measurement covariance 0.01
    x = -35  # Initial estimate
    P = 1  # Initial covariance

    kalman_filters = []
    for i in range(5):
        k = SingleStateKalmanFilter(A, B, C, x, P, Q, R)
        kalman_filters.append(k)

    filtro_output_x = MovingAverageFilter(2)
    filtro_output_y = MovingAverageFilter(2)

    fig, ax = plt.subplots()
    x_plot, y_plot = [0], [0]
    sc = ax.scatter(x_plot, y_plot)
    plt.xlim(0, 1.80)
    plt.ylim(0, 0.90)
    # Major ticks every 20, minor ticks every 5
    # major_ticks_x = np.arange(0, 1.81, 0.30)
    minor_ticks_x = np.arange(0, 1.81, 0.10)
    # major_ticks_y = np.arange(0, 0.91, 0.30)
    minor_ticks_y = np.arange(0, 0.91, 0.10)

    # ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    # ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)
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


    x = 0
    y = 0
    index_rnn = 0
    max_index_rnn = 10

    first_values = True
    c = []
    trajectory = load_trajectory("central_trajectory")
    index_trajcetory = 0
    realtime_rssi_filename = "realtime_data/central_points_rssi.csv"
    while True:
        try:
            if connected < 5:
                print("Waiting connection")
                new_c, addr = s.accept()
                c.append(new_c)
                print('Got connection from', addr)
                connected += 1
                new_client(new_c, addr)
            else:
                time.sleep(1)
                start_valuating = Value('i', True)

                if trajectory:
                    print("Trajectory point", trajectory["points"][index_trajcetory])

                p = []
                for i in range(5):
                    all_data = []
                    for d in dataReader[i]:
                        raw_rssi = dataReader[i].pop(0)
                        kalman_filters[i].step(0, raw_rssi)
                        all_data.append(kalman_filters[i].current_state())
                        # rssi_value[i] = raw_rssi

                    values[i] = kalman_filters[i].current_state()
                    if len(all_data) == 0:
                        kalman_filters[i].step(0, -45)
                        all_data.append(kalman_filters[i].current_state())

                    rssi_value_rnn[i] = max(all_data)
                    rssi_value_cnn[i] = all_data
                    p.append(packet[i])
                    packet[i] = 0

                new_pos = new_evaluete(values)

                print(f"cnn ble_kalman: x", new_pos_cnn[0], "y", new_pos_cnn[1])
                animate(new_pos_cnn[0], new_pos_cnn[1])

                print(f"rnn kalman: x", new_pos_rnn[0], "y", new_pos_rnn[1])
                # animate(new_pos_rnn[0], new_pos_rnn[1])

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
                # animate(filtro_output_x.current_state(), filtro_output_y.current_state())
                # print("Posizione" +" x:"+str(xc)+" y: "+str(yc), flush=True)
                # animate(xc, yc)

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
