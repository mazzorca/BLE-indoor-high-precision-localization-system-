"""
Config to generate some dataset
"""

# if using kalman, the max and min bound
NORM_MAX_K = -15
NORM_MIN_K = -65

# if not using kalman, the max and min bound
NORM_MAX_NK = 0
NORM_MIN_NK = -90


INITIAL_CUT = 10  # initial percentage of each position to cut
FINAL_CUT = 20  # final percentage of each position to cut

WINDOW_SIZE_RNN = 100  # size of one matrix of the rnn
