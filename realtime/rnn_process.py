import time

import numpy as np
import torch
import torchvision
from PIL import Image

import config
from Configuration import dataset_config
from get_from_repeated_tune_search import get_params
from rnns_models import ble
from rnns_testing import load_model as load_model_rnn


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


def worker_evaluate_rnn(n, start_valuating, rssi_value, new_pos_rnn):
    print('Worker: ' + str(n))
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # initialize rnn # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #
    kalman_rnn = "kalman"
    transform_rnn = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])
    params_rnn = {
        "lr": 0.1,
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

    min_k = dataset_config.NORM_MIN_K

    first_time = True
    print("Initialization complete")
    while True:
        time.sleep(1)
        if not start_valuating.value:
            continue

        rnn_rssi_values = []
        for RSSI_value in rssi_value:
            rnn_rssi_values.append((RSSI_value + abs(min_k)))

        if first_time:
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

        new_pos = evaluate_rnn(model_rnn, matrix_np, transform_rnn)
        new_pos_rnn[0] = new_pos[0]
        new_pos_rnn[1] = new_pos[1]