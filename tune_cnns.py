import torch
import torchvision

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial

from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import gc
import os

import Configuration.cnn_config as cnn_conf


def tune_train_model(config, data_dir=None, checkpoint_dir=None):
    model = cnn_conf.MODELS['rfid']['model']
    transform = cnn_conf.MODELS['rfid']['transform']

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            print(torch.cuda.device_count())
            model = torch.nn.DataParallel(model)
    model.to(device)

    print(device)

    dir = config["wxh-stride"]
    dataset = f"{dir}/BLE2605r"

    train_set = RSSIImagesDataset(csv_file=f"{data_dir}/datasets/cnn_dataset/{dataset}/RSSI_images.csv",
                                  root_dir=f"{data_dir}/datasets/cnn_dataset/{dataset}/RSSI_images",
                                  transform=transform)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=config['lr'], momentum=0.9)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

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

    for epoch in range(config['epoch']):
        running_loss = 0.0
        for i, data in enumerate(train_loader):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            # outputs = torch.nn.functional.softmax(outputs, dim=0)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if (i % 10) == 9:
                print(f'[{epoch + 1}, {i + 1}] loss: {running_loss / 10}')
                running_loss = 0.0

        torch.cuda.empty_cache()

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

        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(loss=(val_loss / val_steps), accuracy=correct / total)

    print('Finished Training of:', dir)

    gc.collect()


def main(num_samples, max_num_epochs):
    data_dir = os.path.abspath("./")

    config = {
        "lr": tune.grid_search([0.0001, 0.001, 0.01]),
        "batch_size": tune.grid_search([32, 128, 256]),
        "epoch": tune.grid_search([10, 15, 20]),
        "wxh-stride": tune.grid_search([
            "15x15-10",
            "20x20-10",
            "25x25-10",
            "5x60-10"
        ])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    result = tune.run(
        partial(tune_train_model, data_dir=data_dir),
        resources_per_trial={"cpu": 2, "gpu": 1},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter)

    best_trial = result.get_best_trial("loss", "min", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))


if __name__ == '__main__':
    main(num_samples=1, max_num_epochs=10)
