import torch
import numpy as np
import torchvision
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from rnn_dataset import RnnDataset
from rnns_models import ble

from utility import get_square_number_array
import statistic_utility


if __name__ == '__main__':
    name_file_reader = "dati3105run0r"
    name_file_cam = "Cal3105run0"

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    test_set = RnnDataset(csv_file=f"datasets/rnn_dataset/{name_file_reader}/matrix.csv",
                          root_dir=f"datasets/rnn_dataset/{name_file_reader}",
                          transform=transform)

    test_loader = DataLoader(dataset=test_set,
                             batch_size=1,
                             shuffle=True,
                             num_workers=2)

    model = ble.BLErnn()
    model.load_state_dict(torch.load(f"rnns/ble.pth"))
    model.eval()

    model = model.float()

    optimal_points = np.array([[], []])
    optimal_points = optimal_points.transpose()
    predicted_points = np.array([[], []])
    predicted_points = predicted_points.transpose()
    with torch.no_grad():
        for training_point in test_loader:
            RSSI_matrix, position = training_point[0], training_point[1]

            optimal_points = np.concatenate([optimal_points, np.array(position).reshape(1, 2)])

            position_predicted = model(RSSI_matrix.float())
            predicted_points = np.concatenate([predicted_points, position_predicted.view(2).numpy().reshape(1, 2)])

    euclidean_df = statistic_utility.get_ecdf_euclidean_df(optimal_points, predicted_points, "rnn")
    euclidean_df.plot.line(
        title="ECDF rnn",
        xlabel="(m)",
        ylabel="Empirical cumulative distribution function"
    )

    xo, yo = get_square_number_array(optimal_points[:, 0], optimal_points[:, 1])
    xp, yp = get_square_number_array(predicted_points[:, 0], predicted_points[:, 1])
    square_df = statistic_utility.get_ecdf_square_df(xo, yo, xp, yp, "rnn")
    ax = plt.axes(title="ECDF rnn")
    index = square_df.index.tolist()
    ax.step(np.array(index), square_df['rnn'], label="rnn", where="post")
    plt.legend(loc='lower right')

    plt.show()


