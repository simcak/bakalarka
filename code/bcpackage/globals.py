#################### GLOBAL CONSTANTS ####################
# what sample i do we want to show?
SAMPLE_NUMBER_CB = 0
SAMPLE_NUMBER_BUT = 45

# Tolerance for peak detection (in samples) - for confusion matrix
TOLERANCE = 30

# Quality threshold
CORRELATION_THRESHOLD = 0.7
MORPHO_THRESHOLD = 0.6

# Quality: Morphological thresholds
AMPLITUDE_MIN = 0.5
AMPLITUDE_MAX = 2.0
RISE_TIME_MIN = 0.05
RISE_TIME_MAX = 0.4

##################### GLOBAL ERRORS ######################
INVALID_METHOD = 'Invalid method provided. Use either "my" or "neurokit".'
INVALID_QUALITY_METHOD = 'Invalid method provided. Use either "my" or "orphanidou".'

##################### GLOBAL COLORS ######################
BUT_RED = '#E23F44'
CESA_BLUE = '#02CCFF'

#################### GLOBAL VARIABLES ####################
# File info
CB_FILES = []
CB_FILES_LEN = 0
BUT_DATA_LEN = 0

# Confusion matrix
TP_LIST, FP_LIST, FN_LIST = [], [], []

# 
DIFF_HR_LIST = []
DIFF_HR_LIST_QUALITY = []
QUALITY_LIST = []
DIFF_QUALITY_SUM = 0