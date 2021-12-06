"""
Script to execute the function of visualizer.py, mainly  for visualize data in form of plot
"""
import pandas as pd

import config
import visualizer

if __name__ == "__main__":
    visualize = 2  # change this variable

    if visualize == 0:  # plot the RSSI filtered with kalman filter
        visualizer.plot_kalman_rssi()

    if visualize == 1:  # plot for each reader the value of RSSI and the mean and std
        visualizer.plot_reader_rssi_stats()

    if visualize == 2:  # compare the raw value of RSSI with the value filtered
        visualizer.plot_raw_and_kalman_rssi()

    if visualize == 3:  # compare the value of RSSI filtered with the value extended
        visualizer.plot_kalman_and_extended_kalman_rssi()

    if visualize == 4:  # compare the cut value of RSSI with the value extended
        visualizer.plot_extended_kalman_and_cutted_rssi()

    if visualize == 5:  # Compare the value of RSSI for different parameters of the kalman filter varying in a range
        # with linspace
        visualizer.different_kalman_filter()

    if visualize == 6:  # Compare the value of RSSI for different parameters of the kalman filter
        visualizer.specific_kalman_filter()

    if visualize == 7:  # plot the value of the RSSI without the outliers
        visualizer.plot_dataset_without_outliers()

    if visualize == 8:  # plot the y dataset on the table
        dataset_list = []
        r_list = []
        prefix = "squares/"
        for square_number in range(18):
            name_dataset = f"s{square_number}_2910"
            dataset_list.append(f"{prefix}{name_dataset}")
            r_list.append(f"{prefix}{name_dataset}_r")

        visualizer.plot_y_dataset(name_files_reader=r_list, name_files_cam=dataset_list)

    if visualize == 9:  # plot the number of predicted points varying the parameters in the excel file calculated with
        # the script kalman_parameter_comparator
        visualizer.plot_good_points_line('R', "kpc/kpc-good_pointsRplot.xlsx")

    if visualize == 10:  # plot the hexbin graph of the data obtained with the script kalman_parameter_comparator
        visualizer.plot_good_points_sparse("kpc/kpc-settling_sample-high_range.xlsx", "SETTLING_SAMPLE_AVG")

    if visualize == 11:  # plot the table polygon that can be found in config.py
        visualizer.plot_table_plygons()

    if visualize == 12:  # plot the graph 3d of the study on kalman filter
        visualizer.plot_3d_setting_time_and_predicted_point("kpc/kpc-good_points_high_rangepercentage.xlsx",
                                                            "kpc/kpc-settling-500.xlsx")

    if visualize == 13:  # plot the inference graph with a specific CNN on a specific dataset
        visualizer.cnn_determination_square("ble_kalman/20-0.01-32-20x20-10", "20x20-10", "dati3105run0r")
