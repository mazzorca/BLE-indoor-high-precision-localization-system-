import numpy as np
from matplotlib import pyplot as plt

import utility
import json

if __name__ == '__main__':
    # datiCSV5, datiEMT5 = utility.takeData("dati1410onesquare_r", "tele1410onesquare")
    # utility.printDati(datiCSV5, datiEMT5)
    # print(len(datiCSV5[0]))
    # print(len(datiEMT5[0]))
    #
    # name = "Square15"
    # utility.saveDataArff(datiCSV5, datiEMT5, name)
    # print("fine savedataarff Square15")

    with open(f'near_trajectory.json', ) as f:
        data = json.load(f)

    fig, ax = plt.subplots()

    for point in data["points"]:
        ax.scatter(point['x'], point['y'], c='red')

    plt.xlim(0, 1.80)
    plt.ylim(0, 0.90)

    major_ticks_x = np.arange(0, 1.81, 0.30)
    # minor_ticks_x = np.arange(0, 1.81, 0.10)
    major_ticks_y = np.arange(0, 0.91, 0.30)
    # minor_ticks_y = np.arange(0, 0.91, 0.10)

    ax.set_xticks(major_ticks_x)
    # ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    # ax.set_yticks(minor_ticks_y, minor=True)

    ax.grid(which='both')
    plt.title("Positioning points in the realtime experiment")
    plt.ylabel('Y')
    plt.xlabel('X')

    plt.show()
    plt.savefig("plots/near_trajectory_in_the_table.png")
