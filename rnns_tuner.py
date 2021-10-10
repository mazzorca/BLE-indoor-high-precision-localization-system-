import torch
import torchvision

from torch.utils.data import DataLoader
from torch.utils.data import random_split

import gc
import itertools

import pandas as pd

from rnn_dataset import RnnDataset
from rnns_models import ble
from train_cnns import weight_reset

import random
import numpy as np


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def tune_train_model(config):
    seed = config["trial_num"]
    torch.manual_seed(config["trial_num"])
    g = torch.Generator()
    num_worker = 1
    torch.use_deterministic_algorithms(True)
    g.manual_seed(int(seed))
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    gc.collect()

    model = ble.BLErnn(config["linear_mul"], config["lstm_size"])

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
    model.to(device)

    print(device)

    name_file_reader = "BLE2605r"

    batch_size = 32

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_set = RnnDataset(csv_file=f"datasets/rnn_dataset/{name_file_reader}/matrix.csv",
                           root_dir=f"datasets/rnn_dataset/{name_file_reader}",
                           transform=transform)

    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
        train_set, [test_abs, len(train_set) - test_abs])

    train_loader = DataLoader(dataset=train_subset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              worker_init_fn=seed_worker,
                              generator=g)

    val_loader = DataLoader(dataset=val_subset,
                            batch_size=batch_size,
                            shuffle=True,
                            num_workers=num_worker,
                            worker_init_fn=seed_worker,
                            generator=g)

    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=config["lr"])

    model.apply(weight_reset)

    model = model.float()

    num_trial = config["trial_num"]

    name_run = f"{config['lr']}"
    name_run_with_trial = f"{num_trial}:{name_run}"
    val_loss = 0
    val_acc = 0
    for epoch in range(200):
        print("---------------------------------------------------------")
        print("Epoch:", epoch)

        training_loss = 0.0
        for i, training_point in enumerate(train_loader):
            optimizer.zero_grad()

            RSSI_matrix, position = training_point[0].to(device), training_point[1].to(device)

            position_predicted = model(RSSI_matrix.float())

            loss = loss_function(position_predicted.float(), position.float())
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            # if (i % 10) == 9:
            #     print(f'[{epoch + 1}, {i + 1}] loss: {training_loss / 10}')
            #     training_loss = 0.0

        torch.cuda.empty_cache()
        gc.collect()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        for i, validation_point in enumerate(val_loader, 0):
            with torch.no_grad():
                RSSI_matrix, position = validation_point[0].to(device), validation_point[1].to(device)

                position_predicted = model(RSSI_matrix.float())

                loss = loss_function(position_predicted.float(), position.float())
                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_loss = (val_loss / val_steps)

        print(f"validation loss: {val_loss}")

    print("---------------------------------------------------------")
    print('Finished Training of:', name_run_with_trial)

    print("\n\n")

    return val_loss, val_acc


def expand_grid(config):
    config_list = []
    for key in config.keys():
        config_list.append(config[key])

    config_list = list(itertools.product(*config_list))

    config_dict_list = []
    for config_tuple in config_list:
        config_dict = {}
        for key, elem in zip(config.keys(), config_tuple):
            config_dict[key] = elem
        config_dict_list.append(config_dict)

    return config_dict_list


def main(trials):
    config = {
        "lr": [0.001, 0.01, 0.1],
        "lstm_size": [16, 32, 64, 128],
        "linear_mul": [2, 4, 8]
    }

    list_of_keys = list(config.keys())

    config_dict_list = expand_grid(config)

    df_final = pd.DataFrame()
    for i in range(trials):
        for trial_config in config_dict_list:
            trial_config["trial_num"] = i
            loss, acc = tune_train_model(trial_config)
            trial_config["loss"] = loss
            df_final = df_final.append(trial_config, ignore_index=True)

    df_grouped = df_final.groupby(by=list_of_keys).mean()
    df_grouped.to_csv("rnn.csv")

    df_final.to_csv("rnn_all.csv")


if __name__ == '__main__':
    main(trials=5)
