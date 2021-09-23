import torch
import torchvision

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


def euclidean_pred_and_optimal(preds, ps, probabilities, p):
    xs = p['x'].numpy()
    ys = p['y'].numpy()
    point = np.column_stack([xs, ys])
    ps = np.concatenate([ps, point])

    probabilities_np = probabilities.numpy()
    indexs_np = probabilities_np.argsort()[:, -cnn_conf.NUMBER_ARGMAX_EUCLIDEAN:]

    xs = []
    ys = []
    for index_np, probability_np in zip(indexs_np, probabilities_np):
        normalized_sum = np.sum(probability_np[index_np])

        x = 0
        y = 0
        for index in index_np:
            normalized_probability = probability_np[index]/normalized_sum
            contribution_x = config.SQUARES[index].centroid.x * normalized_probability
            contribution_y = config.SQUARES[index].centroid.y * normalized_probability
            x += contribution_x
            y += contribution_y

        xs.append(x)
        ys.append(y)

    predicted_points = np.column_stack([xs, ys])

    preds = np.concatenate([preds, predicted_points])

    return preds, ps


def square_pred_and_optimal(preds, ys, probabilities, y):
    y = y.numpy()
    ys = np.concatenate([ys, y])

    probabilities_np = probabilities.argmax(1).numpy()
    # probabilities_np = probabilities.numpy().argsort()[:, -cnn_conf.NUMBER_ARGMAX_SQUARE:]

    # reshape_size = batch_size if batch_size < probabilities_np.shape[0] else probabilities_np.shape[0]
    # probabilities_np = probabilities_np.reshape(reshape_size)

    preds = np.concatenate([preds, probabilities_np])
    return preds, ys


def cnn_test(model, dataset, transform, batch_size, type_dist):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    test_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{dir}/{dataset}/RSSI_images.csv",
                                 root_dir=f"datasets/cnn_dataset/{dir}/{dataset}/RSSI_images",
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

            pred = model(X)
            probabilities = torch.nn.functional.softmax(pred, dim=0)

            if type_dist == 0:
                preds, ys = euclidean_pred_and_optimal(preds, ys, probabilities, p)
            if type_dist == 1:
                preds, ys = square_pred_and_optimal(preds, ys, probabilities, y)

    return preds, ys


if __name__ == '__main__':
    wxh = "20x20-10"
    testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
    type_dist = 1

    epoch = 10
    lr = 0.01
    bs = 32

    batch_size = 10

    for model_name in cnn_conf.MODELS:
        if not cnn_conf.active_moodels[model_name]:
            continue

        model = cnn_conf.MODELS[model_name]['model']
        transform = cnn_conf.MODELS[model_name]['transform']
        parameters_saved = "ble/20-0.01-32-5x60-10"
        model = load_model(model, parameters_saved)

        for dataset_name in testing_dataset:
            print(f'testing {model_name} on {dataset_name} dataset')
            preds, ys = cnn_test(model, dataset_name, transform, batch_size, type_dist)
            base_file_name = f'cnn_results/{model_name}/{type_dist}.{epoch}-{lr}-{bs}-{wxh}-{dataset_name}'
            write_cnn_result(base_file_name, preds, ys)
