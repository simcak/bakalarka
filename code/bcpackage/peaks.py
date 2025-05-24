import numpy as np
from . import preprocess

def local_max_detector(signal, min_height, min_distance):
	"""
	Custom function to detect peaks in a signal by comparing the values around with some tresholds.

	Args:
	signal (list or numpy array): The signal to analyze.
	min_height (float): Minimum height to qualify as a peak.
	min_distance (int): Minimum distance between consecutive peaks.

	Returns:
	List of indices of detected peaks.
	"""
	peaks = []
	last_peak_index = -min_distance	# Initialize to a negative number

	for i in range(1, len(signal) - 1):
		# Check if the current point is a peak
		if signal[i - 1] < signal[i] > signal[i + 1] and signal[i] > min_height and i - last_peak_index >= min_distance:
			peaks.append(i)
			last_peak_index = i

	return np.array(peaks)

def detect_peaks(ppg_signal, fs):
	"""
	Detect peaks in the PPG signal.
	"""
	def too_close_peaks(peaks, min_distance):
		if len(peaks) == 0:
			return peaks

		# Remove peaks that are too close to each other
		diffs = np.diff(peaks)
		to_remove = np.where(diffs < min_distance)[0] + 1
		return np.delete(peaks, to_remove)

	# 'time_size' seconds window with 50% overlap
	time_size = 5
	window_size = int(time_size * fs)
	overlap_size = window_size // 2 # floor division
	peaks_detected = []
	# Thresholds
	min_peak_distance = int(fs * 60 / 200)	# num of samples for 200 BPM
	min_peak_height = 0.3

	for start in range(0, len(ppg_signal) - window_size + 1, overlap_size):
		end = start + window_size
		window = ppg_signal[start:end]

		# 1) Rescale/restandardize the window
		standardized_window = preprocess.standardize_normalize_signal(window)

		# 2) Detect peaks in the standardized window
		peaks = local_max_detector(standardized_window, min_peak_height, min_peak_distance)

		# Add ALL the detected peaks to the list for the corresponding window (even doubles)
		peaks_detected.extend(peaks + start)

	# Now we remove duplicates and return
	peaks_detected = np.unique(peaks_detected)

	# Remove peaks that are too close to each other
	peaks_detected = too_close_peaks(peaks_detected, min_peak_distance)

	return peaks_detected