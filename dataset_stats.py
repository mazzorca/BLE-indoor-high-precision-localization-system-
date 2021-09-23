import utility
import dataset_generator
import data_extractor
import config
import data_converter


def get_mean_sample_for_position(name_file_reader, name_file_cam):
    kalman_filter_par = config.KALMAN_BASE
    kalman_data, kalman_time = utility.extract_and_apply_kalman_csv(name_file_reader, kalman_filter_par)
    dati_cam = utility.convertEMT(name_file_cam)
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(kalman_data, kalman_time, dati_cam)
    _, _, index = data_converter.cutReader(dati_reader_fixed, dati_cam, index_cut)

    print(index)

    sum = 0
    for i in range(len(index) - 1):
        value = index[i+1] - index[i]
        sum += value

    avg = sum/(len(index) - 1)

    return avg
