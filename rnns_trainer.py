import torch
import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader

import random

from rnns_models import ble

from rnn_dataset import RnnDataset
from get_from_repeated_tune_search import get_params


use_best_hyper = 1


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def train_rnn(kalman, seed, params):
    g = torch.Generator()
    g.manual_seed(0)
    num_worker = 2
    if seed != -1:
        torch.manual_seed(int(seed))
        torch.use_deterministic_algorithms(True)
        g.manual_seed(int(seed))
        random.seed(int(seed))
        np.random.seed(int(seed))

        num_worker = 1

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    model = ble.BLErnn(int(params["linear_mul"]), int(params["lstm_size"]))

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
    model.to(device)

    name_file_reader = "BLE2605r"
    batch_size = 32

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_set = RnnDataset(csv_file=f"datasets/rnn_dataset/{name_file_reader}/matrix.csv",
                           root_dir=f"datasets/rnn_dataset/{name_file_reader}",
                           transform=transform)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              worker_init_fn=seed_worker,
                              generator=g)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=params["lr"])

    model = model.float()
    for epoch in range(200):
        training_loss = 0.0
        for i, training_point in enumerate(train_loader):
            optimizer.zero_grad()

            RSSI_matrix, position = training_point[0].to(device), training_point[1].to(device)

            position_predicted = model(RSSI_matrix.float())

            loss = loss_function(position_predicted.float(), position.float())
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i % 10) == 9:
                print(f'[{epoch + 1}, {i + 1}] loss: {training_loss / 10}')
                training_loss = 0.0

    torch.save(model.state_dict(), f"rnns/ble_{kalman}.pth")


if __name__ == '__main__':
    params = {
        "lr": 0.01,
        "lstm_size": 32,
        "linear_mul": 4
    }

    kalman = "kalman"

    best_seed = -1
    if use_best_hyper:
        df_params, best_seed = get_params(f"{kalman}/rnn", list(params.keys()))
        for param in params.keys():
            params[param] = df_params.iloc[0][param]

    print("params used:", params)
    print("seed used:", best_seed)

    train_rnn(kalman, best_seed, params)
