import torch
import torchvision

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import gc
import os

import Configuration.cnn_config as cnn_conf


def train_model(model, wxh, dataset, transform, epochs, learning_rate, batch_size, model_name):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    model.to(device)

    print(device)

    train_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{dataset}/RSSI_images.csv",
                                  root_dir=f"datasets/cnn_dataset/{dataset}/RSSI_images",
                                  transform=transform)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=2)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    for epoch in range(epochs):
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

    save_name = f"{model_name}/{epochs}-{learning_rate}-{batch_size}-{wxh}"
    print('Finished Training of:', save_name)

    PATH = f'cnns/{save_name}.pth'
    torch.save(model.state_dict(), PATH)

    gc.collect()


if __name__ == '__main__':
    target_transform = torchvision.transforms.Lambda(
        lambda y: torch.zeros(18, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    batch_size = 64

    for model_name in cnn_conf.MODELS:
        if not cnn_conf.active_moodels[model_name]:
            continue

        model = cnn_conf.MODELS[model_name]['model']
        transform = cnn_conf.MODELS[model_name]['transform']
        train_model(model, "20x20-10", "BLE2605r", transform, batch_size, 0.001, model_name)
