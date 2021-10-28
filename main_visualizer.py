import pandas as pd

import config
import visualizer

if __name__ == "__main__":
    visualize = 13

    if visualize == 0:
        visualizer.plot_kalman_rssi()

    if visualize == 1:
        visualizer.plot_reader_rssi_stats()

    if visualize == 2:
        visualizer.plot_raw_and_kalman_rssi()

    if visualize == 3:
        visualizer.plot_kalman_and_extended_kalman_rssi()

    if visualize == 4:
        visualizer.plot_extended_kalman_and_cutted_rssi()

    if visualize == 5:
        visualizer.different_kalman_filter()

    if visualize == 6:
        visualizer.specific_kalman_filter()

    if visualize == 7:
        visualizer.plot_dataset_without_outliers()

    if visualize == 8:
        visualizer.plot_y_dataset()

    if visualize == 9:
        visualizer.plot_good_points_line('R', "kpc/kpc-good_pointsRplot.xlsx")

    if visualize == 10:
        visualizer.plot_good_points_sparse("kpc/kpc-settling_sample-high_range.xlsx", "SETTLING_SAMPLE_AVG")

    if visualize == 11:
        kalman_filters_dict = {
            'Q': [0.1, 0.000000001],
            'R': [100, 0.000001]
        }

        df = pd.DataFrame(kalman_filters_dict)
        # visualizer.specific_kalman_filter_chunck(df, selected_cut=0)
        visualizer.specific_kalman_filter(df)

    if visualize == 12:
        visualizer.plot_table_plygons()

    if visualize == 13:
        visualizer.plot_3d_setting_time_and_predicted_point("kpc/kpc-good_points_high_rangepercentage.xlsx",
                                                            "kpc/kpc-settling-500.xlsx")

    if visualize == 14:
        visualizer.cnn_determination_square("ble_kalman/20-0.01-32-20x20-10", "20x20-10", "dati3105run0r")
