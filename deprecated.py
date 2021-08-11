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