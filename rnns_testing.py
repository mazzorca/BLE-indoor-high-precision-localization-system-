"""
Script for the testing of the RNN
"""
import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from rnn_dataset import RnnDataset
from rnns_models import ble

from utility import get_square_number_array
import statistic_utility
import utility
from get_from_repeated_tune_search import get_params


def load_model(model, kalman):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(f"rnns/ble_{kalman}.pth", map_location=torch.device(device)))
    return model


def write_rnn_result(base_file_name, preds, ys):
    utility.check_and_if_not_exists_create_folder(base_file_name)

    with open(f"{base_file_name}_p.npy", 'wb') as f:
        np.save(f, preds)

    with open(f"{base_file_name}_o.npy", 'wb') as f:
        np.save(f, ys)


def rnn_test(model, name_file_reader, type_dist):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    test_set = RnnDataset(csv_file=f"datasets/rnn_dataset/{name_file_reader}/matrix.csv",
                          root_dir=f"datasets/rnn_dataset/{name_file_reader}",
                          transform=transform)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)

    model.eval()

    model = model.float()

    optimal_points = np.array([[], []])
    optimal_points = optimal_points.transpose()
    predicted_points = np.array([[], []])
    predicted_points = predicted_points.transpose()
    with torch.no_grad():
        for training_point in test_loader:
            RSSI_matrix, position = training_point[0], training_point[1]

            optimal_points = np.concatenate([optimal_points, np.array(position).reshape(1, 2)])

            position_predicted = model(RSSI_matrix.float())
            predicted_points = np.concatenate([predicted_points, position_predicted.view(2).numpy().reshape(1, 2)])

    if type_dist:
        xo, yo = get_square_number_array(optimal_points[:, 0], optimal_points[:, 1])
        optimal_points = []
        for square_x, square_y in zip(xo, yo):
            square_number = square_y * 6 + square_x
            optimal_points.append(square_number)

        xp, yp = get_square_number_array(predicted_points[:, 0], predicted_points[:, 1])
        predicted_points = []
        for square_x, square_y in zip(xp, yp):
            square_number = square_y * 6 + square_x
            predicted_points.append(square_number)

    return predicted_points, optimal_points


if __name__ == '__main__':
    testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]  # testing datasets
    kalman = "kalman"  # use or not use kalman

    # configuration of the RNN
    params = {
        "lr": 0.01,
        "lstm_size": 32,
        "linear_mul": 4
    }

    df_params, best_seed = get_params(f"{kalman}/rnn", list(params.keys()))
    for param in params.keys():
        params[param] = df_params.iloc[0][param]
        
    model = ble.BLErnn(int(params["linear_mul"]), int(params["lstm_size"]))
    model = load_model(model, kalman)

    for name_file_reader in testing_dataset:
        print("testing:", name_file_reader)
        for type_dist in [0, 1]:
            print(" type:", type_dist)
            preds, ys = rnn_test(model, name_file_reader, type_dist)
            type_dist = f"{type_dist}-1"
            base_file_name = f'cnn_results/rnn_{kalman}/{type_dist}.{name_file_reader}'
            write_rnn_result(base_file_name, preds, ys)

    


