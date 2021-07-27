import utility

if __name__ == "__main__":
    dataset, dataset_time = utility.extract_and_apply_kalman_csv("BLE2605r")
    df_cut = utility.get_index_start_and_end_position(dataset_time)
    chunks = utility.get_chunk(dataset, df_cut)
