import numpy as np
from matplotlib import pyplot as plt

import dataset_generator
import utility
import json

if __name__ == '__main__':
    name_file = "s10_2910_r"
    CAM_FILES_S = ["s10_2910"]

    datiCSV0, datiEMT0 = utility.takeData("BLE2605r", "2605r0")
    datiCSV0_np = np.array(datiCSV0)
    datiEMT0_np = np.array(datiEMT0)
    prefix = "squares/"

    X = [[] for _ in range(5)]
    Y = [[] for _ in range(2)]
    X = np.array(X)
    Y = np.array(Y)
    X = np.concatenate([X, datiCSV0_np], axis=1)
    Y = np.concatenate([Y, datiEMT0_np], axis=1)

    for square_number in range(18):
        print("square:", square_number)
        name_readers = f"s{square_number}_2910"
        datiCSV, datiEMT = utility.takeData(f"{prefix}{name_readers}_r", f"{prefix}{name_readers}")

        X = np.concatenate([X, datiCSV], axis=1)
        Y = np.concatenate([Y, datiEMT], axis=1)

    X = X.transpose()
    Y = Y.transpose()
    dataset_generator.save_dataset_numpy_file("x_train_total", X, "y_train_total", Y)

    name = "Total"
    dataset_generator.save_dataset_numpy_file(f"{name}_RSSI", X, f"{name}_Target", Y)
    utility.saveDataArff(X, Y, name)
    print("fine savedataarff")
