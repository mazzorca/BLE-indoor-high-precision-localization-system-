from kalman import SingleStateKalmanFilter
import csv


# funzione per la conversione dei dati da file .csv dai 5 reader in una lista di array [R1,R2,R3,R4,R5]
# effettuo il filtraggio dei dati con il filtro di kalman e restituisco i dati filtrati e il timestamp
def convertCSV_backup(namefile):
    dataset = []
    datasetTime = []

    '''
    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.00001  # Process covariance 0.00001
    R = 0.01# Measurement covariance 0.01
    x = -35  # Initial estimate
    P = 1  # Initial covariance

    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.0001  # Process covariance 0.0001
    R = 0.5# Measurement covariance 0.5
    x = -35  # Initial estimate
    P = 1  # Initial covariance

    '''
    A = 1  # No process innovation
    C = 1  # Measurement
    B = 1  # No control input
    Q = 0.001  # Process covariance 0.0001
    R = 0.5  # Measurement covariance 0.5
    x = -35  # Initial estimate
    P = 1  # Initial covariance

    kalman_filters = []
    for j in range(5):
        kalman_filter = SingleStateKalmanFilter(A, B, C, x, P, Q, R)
        kalman_filters.append(kalman_filter)

    kalman_filter_estimates = [[], [], [], [], []]

    for i in range(5):
        with open(f"dati/{namefile}{str(i + 1)}.csv") as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=';')
            X = []
            T = []
            line_count = 0
            for row in csv_reader:
                if line_count == 0:
                    # print(f'Column names are {", ".join(row)}')
                    line_count += 1
                else:
                    for col in row:
                        # print(col)
                        t = col.split(',')
                        X.append(float(t[2]))
                        T.append(float(t[1]))
        dataset.append(X)
        # print(len(X))
        datasetTime.append(T)

        for j in range(len(dataset[i])):
            kalman_filters[i].step(0, dataset[i][j])
            kalman_filter_estimates[i].append(kalman_filters[i].current_state())
    # printReader(kalman_filter_estimates,dataset)
    return kalman_filter_estimates, datasetTime


# funziona per il taglio dei dati per via del ritardo che introduce il filtro di kalman
def cutReader_backup(dati, tele, ind):
    newData = [[], [], [], [], []]
    newTele = [[], []]
    new_data = [0, 0, 0, 0, 0]
    old_data = [0, 0, 0, 0, 0]

    for j in range(len(ind) - 1):
        dim = ind[j + 1] - ind[j]
        offset = math.trunc(dim / 3)
        # print(offset)
        lun = offset
        for t in range(lun):
            for i in range(len(dati)):
                if ind[j] + offset + t < len(dati[i]):
                    new_data[i] = dati[i][ind[j] + offset + t]

            if not np.array_equal(new_data, old_data):
                for i in range(len(dati)):
                    newData[i].append(dati[i][ind[j] + offset + t])
                for i in range(5):
                    old_data[i] = new_data[i]
                newTele[0].append(tele[2][ind[j] + offset + t])
                newTele[1].append(tele[3][ind[j] + offset + t])

    # for i in range (5):
    # print(len(newData[i]))
    return newData, newTele


def get_ecdf_dataset_back(x_train, x_test, y_train, y_test, regressors=None):
    if regressors is None:
        regressors = CLASSIFIERS_DICT

    lox, loy = regressors_lib.get_optimal_points(y_test)

    bins = [0.01 * i for i in range(60)]
    ecdf_dict = {}
    for regressor_name in regressors:
        regressors[regressor_name].fit(x_train, y_train)
        Z = regressors[regressor_name].predict(x_test)

        lpr = Z[:, 0]
        lpp = Z[:, 1]
        lpx, lpy = utility.pol2cart(lpr, lpp)

        error_x = abs(np.subtract(lpx, lox))
        error_y = abs(np.subtract(lpy, loy))

        error_x = np.power(error_x, 2)
        error_y = np.power(error_y, 2)
        errors = np.add(error_x, error_y)
        errors = np.sqrt(errors)

        ecdf = [0]
        unit = 1 / len(errors)
        hist = np.histogram(errors, bins)
        cumul_sum = 0
        for i in hist[0]:
            increment = unit * i
            cumul_sum += increment
            ecdf.append(cumul_sum)
        ecdf.append(1)

        ecdf_dict[f'ecdf_{regressor_name}'] = ecdf

    bins = [str(bin_elem) for bin_elem in bins]
    bins.append("0.60+")
    df = pd.DataFrame(ecdf_dict, index=bins)

    return df


def rnns_dataset():
    dati_cam = utility.convertEMT(name_file_cam)
    data, time = data_extractor.get_raw_rssi_csv(name_file_reader)

    min = dataset_config.NORM_MIN_NK

    normalized_data = normalize_rssi(data, min)
    dati_reader_fixed, time_fixed, index_cut = data_converter.fixReader(normalized_data, time, dati_cam)
    index = utility.get_index_start_and_end_position(time_fixed)
    list_of_position = data_converter.transform_in_dataframe(dati_reader_fixed, index)
    dati_cam = [dati_cam[2], dati_cam[3]]
    labels = RSSI_image_converter.get_label(dati_cam, index_cut)

    training_set = []
    windows_size = 100
    stride = 10
    for position_str, label in zip(list_of_position.keys(), labels):
        position_df = list_of_position[position_str]
        for i in range(0, position_df.shape[0] - windows_size, stride):
            sequence_df = position_df.iloc[i:i + windows_size, :]
            training_point = (sequence_df.to_numpy(), [label[1], label[2]])
            training_set.append(training_point)

    training_set = np.array(training_set)
    print(training_set)
    # training_set.reshape(batch_size, 100, 5)
    np.random.shuffle(training_set)