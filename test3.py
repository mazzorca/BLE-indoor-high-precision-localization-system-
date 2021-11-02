import numpy as np
from matplotlib import pyplot as plt

import dataset_generator
import utility
import json

if __name__ == '__main__':
    name_file = "s10_2910_r"
    CAM_FILES_S = ["s10_2910"]

    prefix = "squares/"

    X = [[], [], [], [], []]
    Y = [[], []]
    for square_number in range(18):
        print("square:", square_number)
        name_readers = f"s{square_number}_2910"
        datiCSV, datiEMT = utility.takeData(f"{prefix}{name_readers}_r", f"{prefix}{name_readers}")
        print(len(datiCSV[0]))
        print(len(datiEMT[0]))

        for i, x in enumerate(X):
            x.extend(datiCSV[i])
            print(len(x))

        for i, y in enumerate(Y):
            y.extend(datiEMT[i])
            print(len(y))

    name = "AllSquare"
    dataset_generator.save_dataset_numpy_file(f"{name}_RSSI", X, f"{name}_Target", Y)
    utility.saveDataArff(X, Y, name)
    print("fine savedataarff")
