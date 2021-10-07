import torch
import torchvision

import random
import numpy as np

from RSSI_images_Dataset import RSSIImagesDataset
from torch.utils.data import DataLoader

import gc

import Configuration.cnn_config as cnn_conf
import utility


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def weight_reset(m):
    if isinstance(m, torch.nn.Conv2d) or isinstance(m, torch.nn.Linear):
        m.reset_parameters()


def save_model(model, save_path):
    PATH = f'cnns/{save_path}.pth'
    utility.check_and_if_not_exists_create_folder(PATH)
    torch.save(model.state_dict(), PATH)


def train_model(model, wxh, dataset, transform, epochs, learning_rate, batch_size, model_name, seed=-1, save=True):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
    print(device)

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

    model.to(device)

    train_set = RSSIImagesDataset(csv_file=f"datasets/cnn_dataset/{wxh}/{dataset}/RSSI_images.csv",
                                  root_dir=f"datasets/cnn_dataset/{wxh}/{dataset}/RSSI_images",
                                  transform=transform)

    train_loader = DataLoader(dataset=train_set,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_worker,
                              worker_init_fn=seed_worker,
                              generator=g)

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

    if save:
        save_model(model, save_name)

    gc.collect()

    return model


if __name__ == '__main__':
    # target_transform = torchvision.transforms.Lambda(
    #     lambda y: torch.zeros(18, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    batch_size = 32
    learning_rate = 0.01
    epochs = 20

    for model_name in cnn_conf.MODELS:
        if not cnn_conf.active_moodels[model_name]:
            continue

        model = cnn_conf.MODELS[model_name]['model']
        transform = cnn_conf.MODELS[model_name]['transform']
        train_model(model, "25x25-10", "BLE2605r", transform, epochs, learning_rate, batch_size, model_name)
