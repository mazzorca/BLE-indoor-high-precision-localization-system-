import torch

import utility
from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader

import gc

import numpy as np

import Configuration.cnn_config as cnn_conf
import config


def load_model(model, parameters_saved):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.load_state_dict(torch.load(f"cnns/{parameters_saved}.pth", map_location=torch.device(device)))
    return model


def write_cnn_result(base_file_name, preds, ys):
    utility.check_and_if_not_exists_create_folder(base_file_name)

    with open(f"{base_file_name}_p.npy", 'wb') as f:
        np.save(f, preds)

    with open(f"{base_file_name}_o.npy", 'wb') as f:
        np.save(f, ys)


def get_points_in_xy(probabilities, number_argmax):
    probabilities_np = probabilities.cpu().numpy()
    indexs_np = probabilities_np.argsort()[:, -number_argmax:]

    xs = []
    ys = []
    for index_np, probability_np in zip(indexs_np, probabilities_np):
        normalized_sum = np.sum(probability_np[index_np])

        x = 0
        y = 0
        for index in index_np:
            normalized_probability = probability_np[index] / normalized_sum
            contribution_x = config.SQUARES[index].centroid.x * normalized_probability
            contribution_y = config.SQUARES[index].centroid.y * normalized_probability
            x += contribution_x
            y += contribution_y

        xs.append(x)
        ys.append(y)

    return xs, ys


def euclidean_pred_and_optimal(preds, ps, probabilities, p, number_argmax):
    xs = p['x'].numpy()
    ys = p['y'].numpy()
    point = np.column_stack([xs, ys])
    ps = np.concatenate([ps, point])

    xs, ys = get_points_in_xy(probabilities, number_argmax)
    predicted_points = np.column_stack([xs, ys])

    preds = np.concatenate([preds, predicted_points])

    return preds, ps


def square_pred_and_optimal(preds, ys, probabilities, y, number_argmax):
    y = y.cpu().numpy()
    ys = np.concatenate([ys, y])

    if number_argmax == 1:
        square_numbers = probabilities.argmax(1).cpu().numpy()
    else:
        xs_point, ys_point = get_points_in_xy(probabilities, number_argmax)
        squares_x, squares_y = utility.get_square_number_array(xs_point, ys_point)

        square_numbers = []
        for square_x, square_y in zip(squares_x, squares_y):
            square_number = square_y * 6 + square_x
            square_numbers.append(square_number)

    preds = np.concatenate([preds, square_numbers])

    return preds, ys


def cnn_test(model, wxh, dataset, transform, batch_size, type_dist, number_argmax=cnn_conf.NUMBER_ARGMAX_EUCLIDEAN):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    test_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{wxh}/{dataset}/RSSI_images.csv",
                                 root_dir=f"datasets/cnn_dataset/{wxh}/{dataset}/RSSI_images",
                                 transform=transform)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=2)

    model.eval()
    model.to(device)

    ys = np.array([])
    preds = np.array([])
    if type_dist == 0:
        ys = np.array([[], []])
        ys = ys.transpose()
        preds = np.array([[], []])
        preds = preds.transpose()

    with torch.no_grad():
        for X, y, p in test_loader:
            X, y = X.to(device), y.to(device)

            probabilities = model(X)
            probabilities = torch.nn.functional.softmax(probabilities, dim=1)

            if type_dist == 0:
                preds, ys = euclidean_pred_and_optimal(preds, ys, probabilities, p, number_argmax)
            if type_dist == 1:
                preds, ys = square_pred_and_optimal(preds, ys, probabilities, y, number_argmax)

    return preds, ys


def test_accuracy(model, transform, wxh, batch_size):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    dataset = f"{wxh}/dati3105run2r"
    test_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{dataset}/RSSI_images.csv",
                                 root_dir=f"datasets/cnn_dataset/{dataset}/RSSI_images",
                                 transform=transform)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=batch_size,
                             shuffle=True,
                             num_workers=8)

    test_steps = 0
    total = 0
    correct = 0
    for i, data in enumerate(test_loader, 0):
        with torch.no_grad():
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            test_steps += 1

    accuracy = correct / total

    return accuracy


if __name__ == '__main__':
    wxh = "20x20-10"
    testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    type_dist = 1

    epoch = 10
    lr = 0.001
    bs = 32

    batch_size = 10

    for model_name in cnn_conf.MODELS:
        if not cnn_conf.active_moodels[model_name]:
            continue

        model = cnn_conf.MODELS[model_name]['model']
        transform = cnn_conf.MODELS[model_name]['transform']
        parameters_saved = "ble/10-0.001-32-20x20-10"
        model = load_model(model, parameters_saved)

        for dataset_name in testing_dataset:
            print(f'testing {model_name} on {dataset_name} dataset')
            preds, ys = cnn_test(model, wxh, dataset_name, transform, batch_size, type_dist)
            base_file_name = f'cnn_results/{model_name}/{type_dist}.{epoch}-{lr}-{bs}-{wxh}-{dataset_name}'
            write_cnn_result(base_file_name, preds, ys)
