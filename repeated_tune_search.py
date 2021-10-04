import torch
import torchvision

from torch.utils.tensorboard import SummaryWriter

from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import gc
import os
import itertools

import pandas as pd

import Configuration.cnn_config as cnn_conf
from train_cnns import weight_reset


model_name = 'resnet50'


def tune_train_model(config, writer):
    torch.manual_seed(config["trial_num"])

    gc.collect()

    model = cnn_conf.MODELS[model_name]['model']
    transform = cnn_conf.MODELS[model_name]['transform']

    model.apply(weight_reset)

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
    model.to(device)

    print(device)

    wxh = config["wxh-stride"]
    dataset = f"{wxh}/BLE2605r"

    train_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{dataset}/RSSI_images.csv",
                                  root_dir=f"datasets/cnn_dataset/{dataset}/RSSI_images",
                                  transform=transform)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    test_abs = int(len(train_set) * 0.8)
    train_subset, val_subset = random_split(
        train_set, [test_abs, len(train_set) - test_abs])

    train_loader = DataLoader(dataset=train_subset,
                              batch_size=config['batch_size'],
                              shuffle=True,
                              num_workers=8)

    val_loader = DataLoader(dataset=val_subset,
                            batch_size=config['batch_size'],
                            shuffle=True,
                            num_workers=8)

    max_epochs = config['epoch']
    num_trial = config["trial_num"]

    name_run = f"{config['lr']}.{config['batch_size']}.{max_epochs}.{wxh}"
    name_run_with_trial = f"{num_trial}:{name_run}"
    val_loss = 0
    val_acc = 0
    for epoch in range(max_epochs):
        print("---------------------------------------------------------")
        print("Epoch:", epoch)

        running_loss = 0.0
        running_steps = 0

        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # outputs = torch.nn.functional.softmax(outputs, dim=0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            running_steps += 1

        train_loss = running_loss/running_steps
        print(f"Training loss:", train_loss)

        torch.cuda.empty_cache()
        gc.collect()

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        correct = 0
        for i, data in enumerate(val_loader, 0):
            with torch.no_grad():
                inputs, labels = data[0].to(device), data[1].to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                loss = criterion(outputs, labels)
                val_loss += loss.cpu().numpy()
                val_steps += 1

        val_loss = (val_loss / val_steps)
        val_acc = correct / total

        print(f"validation loss: {val_loss}, accuracy: {val_acc}")

        writer.add_scalars(f"{name_run_with_trial} loss", 
                           {
                                'val_loss': val_loss,
                                'train_loss': train_loss,
                           }, epoch)

        writer.add_scalar(f"{name_run} accuracy", val_acc, epoch)

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
        "lr": [0.0001, 0.001, 0.01],
        "batch_size": [32, 64, 128],
        "epoch": [10, 15, 20],
        "wxh-stride": [
            "15x15-10",
            "20x20-10",
            "25x25-10",
            "5x60-10"
        ]
    }

    list_of_keys = list(config.keys())

    config_dict_list = expand_grid(config)

    writer = SummaryWriter()
    df_final = pd.DataFrame()
    for i in range(trials):
        for trial_config in config_dict_list:
            trial_config["trial_num"] = i
            loss, acc = tune_train_model(trial_config, writer)
            trial_config["loss"] = loss
            trial_config["acc"] = acc
            df_final = df_final.append(trial_config, ignore_index=True)

    df_grouped = df_final.groupby(by=list_of_keys).mean()
    df_grouped.to_csv(f"{model_name}.csv")
    df_final.to_csv(f"{model_name}_all.csv")


if __name__ == '__main__':
    main(trials=5)
