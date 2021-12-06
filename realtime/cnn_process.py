"""
Script that contains the body of the cnn process in the real time experiment

For the cnn, the values of the RSSI collected by the process in a slice of time, will be added ad the end of the
current matrix.
The matrix is initialized with a  copy of the first value
"""
import itertools
import time

import numpy as np
import torch
from PIL import Image
from skimage import io

import config
from Configuration import cnn_config, dataset_config
from cnn_testing import load_model as load_model_cnn


def evaluate_cnn(model_cnn, image_np, transform):
    image = io.imread("predict_img.jpg")
    img = Image.fromarray(image).convert('RGB')

    # img = Image.fromarray(image_np, 'L')
    # img = img.convert('RGB')

    tensor_img = transform(img)

    # tensor_img = tensor_img.view([1, 1, 24, 24])
    tensor_img = tensor_img[None, :, :, :]

    with torch.no_grad():
        pred = model_cnn(tensor_img)
        probability = torch.nn.functional.softmax(pred, dim=1)  # dare un occhiata

        probability_np = probability.cpu().numpy()[0]

        indexs = probability_np.argsort()[-18:]

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


def worker_evaluate_cnn(n, start_valuating, rssi_value, new_pos_cnn):
    print('Worker: ' + str(n))
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

    # kalman_filter_par = config.KALMAN_BASE
    # kalman_filter_cnn = data_converter.create_kalman_filter(kalman_filter_par)
    # - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

    image_np = np.zeros((20, 20))

    h = 4
    w = 20
    first_time = True
    print("Initialization complete")
    while True:
        time.sleep(1)
        if not start_valuating.value:
            continue

        min_k = dataset_config.NORM_MIN_K
        max_k = dataset_config.NORM_MAX_K

        dividend = min_k - max_k

        rssi_value_copy = [elem for elem in rssi_value]

        max_dim = max([len(elem) for elem in rssi_value_copy])

        for vec in rssi_value_copy:
            vec.extend([vec[-1]] * (max_dim - len(vec)))

        new_cnn_rssi_values = np.array(rssi_value_copy).transpose()
        for rssi_row in new_cnn_rssi_values:
            cnn_rssi_values = []
            for RSSI_value in rssi_row:
                cnn_rssi_values.append(((RSSI_value - max_k) / dividend) * 255)

            if first_time:
                init_rssi_values = list(itertools.chain(cnn_rssi_values, cnn_rssi_values, cnn_rssi_values, cnn_rssi_values))
                image_np = np.full((h * 5, w), init_rssi_values, dtype=np.uint8).transpose()
                if False:
                    image_np_predict = np.array([image_np, image_np, image_np])
                    img = Image.fromarray(image_np_predict, 'RGB')
                    # img = img.convert('RGB')
                    img.show()
                first_time = False

            new_values_rssi = np.array(cnn_rssi_values, dtype=np.uint8)
            image_np = np.concatenate([image_np.flatten('F')[5:], new_values_rssi]).reshape(h * 5, w, order='F')

            if config.debug_cnn:
                img = Image.fromarray(image_np_predict, 'RGB')
                # img = img.convert('RGB')
                img.show()

        img = Image.fromarray(image_np, 'L')
        img.save("predict_img.jpg")

        new_pos = evaluate_cnn(model_cnn, image_np, transform_cnn)
        new_pos_cnn[0] = new_pos[0]
        new_pos_cnn[1] = new_pos[1]
