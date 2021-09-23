import torch
import torchvision

import utility
from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader

import gc

import numpy as np

import Configuration.cnn_config as cnn_conf
import config


dir = "20x20-10"
testing_dataset = ["dati3105run0r", "dati3105run1r", "dati3105run2r"]
type_dist = 1


def write_cnn_result(model_name, dataset, preds, ys):
    base_file_name = f'cnn_results/{dir}/{type_dist}.{model_name}-{dataset}'
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
    ys = np.concatenate([ys, y])

    probabilities_np = probabilities.numpy().argsort()[:, -cnn_conf.NUMBER_ARGMAX_SQUARE:]

    reshape_size = batch_size if batch_size < probabilities_np.shape[0] else probabilities_np.shape[0]
    probabilities_np = probabilities_np.reshape(reshape_size)
    preds = np.concatenate([preds, probabilities_np])

    return preds, ys


def cnn_test(model, dataset, transform, batch_size, device):
    test_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{dataset}/RSSI_images.csv",
                                 root_dir=f"datasets/cnn_dataset/{dataset}/RSSI_images",
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
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 10

    for model_name in cnn_conf.MODELS:
        if not cnn_conf.active_moodels[model_name]:
            continue

        model = cnn_conf.MODELS[model_name]['model']
        transform = cnn_conf.MODELS[model_name]['transform']
        model.load_state_dict(torch.load(f"cnns/{model_name}.pth", map_location=torch.device(device)))

        for dataset_name in testing_dataset:
            print(f'testing {model_name} on {dataset_name} dataset')
            preds, ys = cnn_test(model, dataset_name, transform, batch_size, device)
            write_cnn_result(model_name, dataset_name, preds, ys)
