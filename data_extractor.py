"""
The files contains all the functions needed to extract the data from the files
"""

import csv


def get_raw_rssi_csv_reader(name_file):
    """
    Get the raw rssi value from the file for a single reader
    :param name_file: Name of the file where there is the data
    :return: Two list, one with the rssi value, and another with the time for that rssi
    """
    raw_data = []
    raw_time = []

    with open(name_file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0

        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue

            raw_data.append(float(row[2]))
            raw_time.append(float(row[1]))

    return raw_data, raw_time


def get_raw_rssi_csv(name_file):
    """
    Get the raw rssi value for all reader
    :param name_file: Name of the file where there is the data
    :return: Two list of lists, one with the rssi value, and another with the time for that rssi
    """
    raws_data = []
    raws_time = []

    for i in range(5):
        raw_data, raw_time = get_raw_rssi_csv_reader(f"dati/{name_file}{str(i + 1)}.csv")
        raws_data.append(raw_data)
        raws_time.append(raw_time)

    return raws_data, raws_time
