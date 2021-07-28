import utility
import numpy as np


def generate_dataset_from_a_kalman_data(kalman_data, time, name_file_cam):
    dati_cam = utility.convertEMT(name_file_cam)
    dati_reader_fixed, time_fixed, index_cut = utility.fixReader(kalman_data, time, dati_cam)
    dati_reader_cut, dati_cam_cut = utility.cutReader(dati_reader_fixed, dati_cam, index_cut)

    X = np.array(dati_reader_cut)
    X = np.transpose(X)

    cam_cut_np = np.array(dati_cam_cut)
    cam_cut_np = np.transpose(cam_cut_np)

    ry, py = utility.cart2pol(cam_cut_np[:, 0], cam_cut_np[:, 1])
    y = np.column_stack([ry, py])

    return X, y


if __name__ == "__main__":
    datiCSV1, datiEMT1 = utility.takeData("dati3105run0r", "Cal3105run0")
    utility.printDati(datiCSV1, datiEMT1)
    print(len(datiCSV1[0]))
    print(len(datiEMT1[0]))

    datiCSV2, datiEMT2 = utility.takeData("dati3105run1r", "Cal3105run1")
    utility.printDati(datiCSV2, datiEMT2)
    print(len(datiCSV2[0]))
    print(len(datiEMT2[0]))

    datiCSV3, datiEMT3 = utility.takeData("dati3105run2r", "Cal3105run2")
    utility.printDati(datiCSV3, datiEMT3)
    print(len(datiCSV3[0]))
    print(len(datiEMT3[0]))

    datiCSV0, datiEMT0 = utility.takeData("BLE2605r", "2605r0")
    utility.printDati(datiCSV0, datiEMT0)
    print(len(datiCSV0[0]))
    print(len(datiEMT0[0]))

    name = "Train"
    utility.saveDataArff(datiCSV0, datiEMT0, name)
    print("fine savedataarff 0")

    name = "Test1"
    utility.saveDataArff(datiCSV1, datiEMT1, name)
    print("fine savedataarff 1")

    name = "Test2"
    utility.saveDataArff(datiCSV2, datiEMT2, name)
    print("fine savedataarff 2")

    name = "Test3"
    utility.saveDataArff(datiCSV3, datiEMT3, name)
    print("fine savedataarff 3")

    X = [[], [], [], [], []]
    Y = [[], []]
    for i, x in enumerate(X):
        x.extend(datiCSV0[i])
        x.extend(datiCSV3[i])
        x.extend(datiCSV2[i])
        x.extend(datiCSV1[i])
        print(len(x))

    for i, y in enumerate(Y):
        y.extend(datiEMT0[i])
        Y.extend(datiEMT3[i])
        y.extend(datiEMT2[i])
        y.extend(datiEMT1[i])
        print(len(y))

    name = "Train0"
    utility.saveDataArff(X, Y, name)
    print("fine savedataarff 00")
