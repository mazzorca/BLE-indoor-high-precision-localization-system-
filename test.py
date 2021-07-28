import utility
import dataset_generator
import numpy as np

import testMultiRegress

if __name__ == "__main__":
    # dataset, dataset_time = utility.extract_and_apply_kalman_csv("BLE2605r")
    # reader_cut, cam_cut = dataset_generator.generate_dataset_from_a_kalman_data(dataset, dataset_time, "2605r0")

    testMultiRegress.performance_dataset()
