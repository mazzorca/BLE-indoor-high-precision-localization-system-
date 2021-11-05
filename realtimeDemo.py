import socket
import multiprocessing
from multiprocessing import Process, Manager, Value
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
import paramiko as paramiko

import utility

from kalman import SingleStateKalmanFilter

from realtime.cnn_process import worker_evaluate_cnn
from realtime.rnn_process import worker_evaluate_rnn
from realtime.regressor_process import worker_evaluate_regressor

DIM_REALTIME = 5

readers = ["192.168.1.2",
           "192.168.1.29",
           "192.168.1.48",
           "192.168.1.6",
           "192.168.1.26"]
username = "pi"
password = "raspberry"

ipclient = ["192.168.1.2", "192.168.1.29", "192.168.1.48", "192.168.1.6", "192.168.1.26"]


def new_client(clientsocket, addr, dataReader, packet):
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


def initialize_reader_csv(type_run):
    base_name = f"realtime_data/rssi_{type_run}_r"

    for i in range(5):
        utility.append_to_csv(f"{base_name}{i}.csv", [[f"RSSI", "time"]])


def write_reader_rssi(index, type_run, rssi_list):
    base_name = f"realtime_data/rssi_{type_run}_r"

    utility.append_to_csv(f"{base_name}{index}.csv", rssi_list)


def realtime_process(n, ready, auto_launch=False, config=None):
    print('Process: ' + str(n))

    plt.ion()

    s = socket.socket()
    port = 5000

    manager = Manager()

    dataReader = manager.list([
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45]),
        manager.list([-45])
    ])

    packet = manager.list()
    for i in range(5):
        packet.append(0)

    start_valuating = Value('i', False)


    rssi_value_cnn = manager.list([
        [-45],
        [-45],
        [-45],
        [-45],
        [-45]
    ])
    new_pos_cnn = manager.list([0, 0])

    if config["CNN"]:
        proc_cnn = Process(target=worker_evaluate_cnn, args=("cnn", start_valuating, rssi_value_cnn, new_pos_cnn,))
        proc_cnn.start()

    rssi_value_rnn = manager.list([-45, -45, -45, -45, -45])
    new_pos_rnn = manager.list([0, 0])

    if config["RNN"]:
        proc_rnn = Process(target=worker_evaluate_rnn, args=("rnn", start_valuating, rssi_value_rnn, new_pos_rnn,))
        proc_rnn.start()

    rssi_value_regr = manager.list([-45, -45, -45, -45, -45])
    new_pos_regr = manager.list([
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0],
        [0, 0]
    ])

    if config["Regressor"]:
        proc_regr = Process(target=worker_evaluate_regressor,
                            args=("regressor", start_valuating, rssi_value_regr, new_pos_regr,))
        proc_regr.start()

    host = "192.168.1.41"
    connected = 0
    threads = []

    s.bind((host, port))  # Bind to the port
    s.listen(5)  # Now wait for client connection.
    print('Server started!')

    values = [0.0, 0.0, 0.0, 0.0, 0.0]


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

    fig, ax = plt.subplots()
    x_plot = [
        [0 for _ in range(DIM_REALTIME)],
        [0 for _ in range(DIM_REALTIME)],
        [0 for _ in range(DIM_REALTIME)]
    ]
    y_plot = [
        [0 for _ in range(DIM_REALTIME)],
        [0 for _ in range(DIM_REALTIME)],
        [0 for _ in range(DIM_REALTIME)]
    ]
    c_list = [i for i in range(DIM_REALTIME)]
    sc = [ax.scatter(x_plot[0], y_plot[0], s=250, c=c_list, cmap="Blues", label="CNN"),
          ax.scatter(x_plot[1], y_plot[1], s=250, c=c_list, cmap="Greens", label="k-NN"),
          ax.scatter(x_plot[2], y_plot[2], s=250, c=c_list, cmap="Reds", label="RNN")]

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
    plt.legend()

    legend = ax.get_legend()
    legend.legendHandles[0].set_color(plt.cm.Blues(.8))
    legend.legendHandles[1].set_color(plt.cm.Greens(.8))
    legend.legendHandles[2].set_color(plt.cm.Reds(.8))

    def animate(xt, yt, index):
        x_plot[index].append(xt)
        y_plot[index].append(yt)

        if len(x_plot[index]) > DIM_REALTIME:
            x_plot[index].pop(0)
            y_plot[index].pop(0)
        sc[index].set_offsets(np.c_[x_plot[index], y_plot[index]])
        plt.show(block=False)
        plt.pause(0.001)

    x = 0
    y = 0
    index_rnn = 0
    max_index_rnn = 10

    first_values = True
    c = []
    trajectory = load_trajectory("endless")
    index_trajcetory = 0

    names = ["Random forest", "Nearest Neighbors U", "Nearest Neighbors D", "Decision Tree", "tot"]
    type_run = trajectory["type_run"]
    realtime_regressor_pos_filename = {}
    for name in names:
        realtime_regressor_pos_filename[name] = f"realtime_data/{name}_pos_{type_run}.csv"
    realtime_cnn_pos_filename = f"realtime_data/cnn_pos_{type_run}.csv"
    realtime_rnn_pos_filename = f"realtime_data/rnn_pos_{type_run}.csv"

    if trajectory["save"]:
        initialize_reader_csv(type_run)
        for name in names:
            utility.append_to_csv(realtime_regressor_pos_filename[name], [["x", "y", "time"]])

        utility.append_to_csv(realtime_cnn_pos_filename, [["x", "y", "time"]])
        utility.append_to_csv(realtime_rnn_pos_filename, [["x", "y", "time"]])

    start = False
    sec_left = trajectory["points"][0]["time"]
    deltaT = trajectory["deltaT"]
    transition_timer = 0
    transition = False
    if n != "single":
        ready.value = True
    if auto_launch:
        for reader in readers:
            print("reader")
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(reader, username=username, password=password)
            command = "sudo iwconfig wlan0 power off; cd Desktop/BLE-Beacon-Scanner; python3 BeaconScanner.py"
            ssh_stdin, ssh_stdout, ssh_stderr = ssh.exec_command(command)

    while True:
        try:
            if connected < 5:
                print("Waiting connection")
                new_c, addr = s.accept()
                c.append(new_c)
                print('Got connection from', addr)
                connected += 1
                new_client(new_c, addr, dataReader, packet)
            else:
                time.sleep(0.5)
                if not start:
                    command = input("Press s to start the realtime")
                    if command == 's':
                        start = True
                    continue

                start_valuating = Value('i', True)

                if transition:
                    os.system(f'say "{transition_timer}"')

                if transition_timer == 0 and transition:
                    if index_trajcetory < len(trajectory["points"]):
                        index_trajcetory += 1
                    else:
                        break
                    sec_left = trajectory["points"][index_trajcetory]["time"]
                    transition = False

                if sec_left == trajectory["points"][index_trajcetory]["time"]:
                    os.system(f'say "start position {index_trajcetory}"')

                if sec_left == 0 and not transition:
                    transition_timer = deltaT
                    os.system(f'say "start transition"')
                    transition = True

                if trajectory:
                    if transition:
                        print("Transitioning")
                    else:
                        print("Trajectory point", trajectory["points"][index_trajcetory])

                p = []

                for i in range(5):
                    all_data = []
                    raw_rssi_list = []
                    raw_rssi_only = []
                    for d in dataReader[i]:
                        raw_rssi = dataReader[i].pop(0)
                        ts = time.time()
                        raw_rssi_list.append([raw_rssi, ts])
                        raw_rssi_only.append(raw_rssi)

                        kalman_filters[i].step(0, raw_rssi)
                        all_data.append(kalman_filters[i].current_state())
                        # rssi_value[i] = raw_rssi

                    if trajectory["save"]:
                        write_reader_rssi(i, type_run, raw_rssi_list)

                    values[i] = kalman_filters[i].current_state()
                    rssi_value_regr[i] = values[i]
                    if len(all_data) == 0:
                        kalman_filters[i].step(0, -45)
                        all_data.append(kalman_filters[i].current_state())

                    if len(raw_rssi_only) == 0:
                        raw_rssi_only.append(-45)

                    rssi_value_rnn[i] = max(all_data)
                    rssi_value_cnn[i] = all_data
                    p.append(packet[i])
                    packet[i] = 0

                ts = time.time()

                if config["CNN"]:
                    x_cnn = new_pos_cnn[0]
                    y_cnn = new_pos_cnn[1]
                    print(f"cnn resnet50 no_kalman: x", x_cnn, "y", y_cnn)
                    if trajectory["save"]:
                        utility.append_to_csv(realtime_cnn_pos_filename, [[x_cnn, y_cnn, ts]])
                    animate(x_cnn, y_cnn, 0)

                if config["RNN"]:
                    x_rnn = new_pos_rnn[0]
                    y_rnn = new_pos_rnn[1]
                    print(f"rnn nokalman: x", x_rnn, "y", y_rnn)
                    if trajectory["save"]:
                        utility.append_to_csv(realtime_rnn_pos_filename, [[x_rnn, y_rnn, ts]])
                    animate(x_rnn, y_rnn, 2)

                if config["Regressor"]:
                    for rn, file_name in enumerate(realtime_regressor_pos_filename.keys()):
                        x = new_pos_regr[rn][0]
                        y = new_pos_regr[rn][1]
                        print(names[rn], "x:", x, "y:", y, flush=True)

                        if trajectory["save"]:
                            utility.append_to_csv(realtime_regressor_pos_filename[file_name], [[x, y, ts]])

                        if rn == 2:
                            animate(x, y, 1)

                print("Pacchetti raccolti:", p[0], p[1], p[2], p[3], p[4], flush=True)

                for i in range(5):
                    values[i] = kalman_filters[i].current_state()
                print("RSSI:", values[0], values[1], values[2], values[3], values[4], flush=True)

                print("\n")

                first_values = False
                if transition:
                    transition_timer -= 1
                else:
                    sec_left -= 1
        except KeyboardInterrupt:
            for connection in c:
                if connection:
                    connection.close()
            break


def seletion_demo():
    config = {
        "Regressor": False,
        "CNN": False,
        "RNN": False
    }
    print("Select the solutions to adopt:\n 1: regressors\n 2: CNN\n 3. RNN")
    solutions = input("?> ")

    if "1" in solutions:
        config["Regressor"] = True
    if "2" in solutions:
        config["CNN"] = True
    if "3" in solutions:
        config["RNN"] = True

    return config


if __name__ == '__main__':
    config = seletion_demo()
    multiprocessing.freeze_support()
    realtime_process("single", True, True, config=config)
