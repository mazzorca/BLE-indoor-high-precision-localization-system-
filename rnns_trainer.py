import torch
import numpy as np
import torchvision.transforms
from torch.utils.data import DataLoader

from rnns_models import ble

from rnn_dataset import RnnDataset

if __name__ == '__main__':
    name_file_reader = "BLE2605r"
    name_file_cam = "2605r0"

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
                              num_workers=2)

    model = ble.BLErnn()
    loss_function = torch.nn.MSELoss()
    optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)

    model = model.float()
    for epoch in range(200):
        training_loss = 0.0
        for i, training_point in enumerate(train_loader):
            optimizer.zero_grad()

            RSSI_matrix, position = training_point[0], training_point[1]

            position_predicted = model(RSSI_matrix.float())

            loss = loss_function(position_predicted.float(), position.float())
            training_loss += loss.item()
            loss.backward()
            optimizer.step()

            if (i % 10) == 9:
                print(f'[{epoch + 1}, {i + 1}] loss: {training_loss / 10}')
                training_loss = 0.0

    torch.save(model.state_dict(), "rnns/ble.pth")

    name_file_reader = "dati3105run2r"
    name_file_cam = "Cal3105run2"

    test_set = RnnDataset(csv_file=f"datasets/rnn_dataset/{name_file_reader}/matrix.csv",
                          root_dir=f"datasets/rnn_dataset/{name_file_reader}",
                          transform=transform)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)

    optimal_points = np.array([[], []])
    optimal_points = optimal_points.transpose()
    predicted_points = np.array([[], []])
    predicted_points = predicted_points.transpose()
    with torch.no_grad():
        for testing_point in test_loader:
            RSSI_matrix, position = testing_point[0], testing_point[1]

            optimal_points = np.concatenate([optimal_points, position.numpy().reshape(1, 2)])

            position_predicted = model(RSSI_matrix.float())
            predicted_points = np.concatenate([predicted_points, position_predicted.view(2).numpy().reshape(1, 2)])

    print(predicted_points.max(axis=0))
    print(predicted_points.min(axis=0))
