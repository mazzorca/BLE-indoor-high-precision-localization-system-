"""
Contains all the constants used project wide
"""

NAME_FILES = ["BLE2605r", "dati3105run0r", "dati3105run1r", "dati3105run2r"]
CAM_FILES = ["2605r0", "Cal3105run0", "Cal3105run1", "Cal3105run2"]

N_NEIGHBOURS = 100
CORD = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
CORD_X = [15, 45, 75, 105, 135, 165, 15, 45, 75, 105, 135, 165, 15, 45, 75, 105, 135, 165]
CORD_Y = [75, 75, 75, 75, 75, 75, 45, 45, 45, 45, 45, 45, 15, 15, 15, 15, 15, 15]

OUTLIERS_BAND = 1.
MAX_T_ASS_SAMPLES = 100

OPTIMAL_TH = 0.05

KALMAN_BASE = {
        'A': 1.,  # No process innovation
        'C': 1.,  # Measurement
        'B': 1.,  # No control input
        'Q': 0.00001,  # Process covariance 0.0001
        'R': 0.01,  # Measurement covariance 0.5
        'x': -35.,  # Initial estimate
        'P': 1.  # Initial covariance
}

KALMAN_1 = {
        'A': 1.,  # No process innovation
        'C': 1.,  # Measurement
        'B': 1.,  # No control input
        'Q': 0.00002,  # Process covariance 0.0001
        'R': 0.025,  # Measurement covariance 0.5
        'x': -35.,  # Initial estimate
        'P': 1.  # Initial covariance
}
