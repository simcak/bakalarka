# what sample id of the chosen database do we want to show?
SAMPLE_NUMBER_CB = 26
SAMPLE_NUMBER_BUT = 45

# Quality threshold
CORRELATION_THRESHOLD = 0.7

# Quality: Morphological thresholds
AMPLITUDE_MIN = 0.1
AMPLITUDE_MAX = 1.0
RISE_TIME_MIN = 0.05
RISE_TIME_MAX = 0.4

# Error
INVALID_METHOD = 'Invalid method provided. Use either "my" or "neurokit".'
INVALID_QUALITY_METHOD = 'Invalid method provided. Use either "my" or "orphanidou".'

# File info
CB_FILES = []
CB_FILES_LEN = 0
BUT_DATA_LEN = 0

# Lists for storing TP, FP, FN
TP_LIST, FP_LIST, FN_LIST = [], [], []

# 
DIFF_HR_LIST = []
QUALITY_LIST = []
DIFF_QUALITY_SUM = 0