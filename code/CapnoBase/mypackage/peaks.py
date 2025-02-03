import numpy as np

def bc_find_peaks(signal, min_height, min_distance):
	"""
	Custom function to detect peaks in a signal.
	
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
		if signal[i - 1] < signal[i] > signal[i + 1] and signal[i] > min_height:
			# Ensure the peak is at least `min_distance` away from the last detected peak
			if i - last_peak_index >= min_distance:
				peaks.append(i)
				last_peak_index = i

	return np.array(peaks)

def detect_peaks(ppg_signal, capnobase_fs):
	"""
	Detect peaks in the PPG signal.
	"""
	# 10 seconds window with 50% overlap
	window_size = int(10 * capnobase_fs)
	overlap_size = window_size // 2
	peaks_detected = []

	for start in range(0, len(ppg_signal) - window_size + 1, overlap_size):
		end = start + window_size
		window = ppg_signal[start:end]

		# 1) Rescale/restandardize the window
		window_mean = np.mean(window)
		window_std = np.std(window)
		if (window_std == 0):
			continue
		standardized_window = (window - window_mean) / window_std

		# 2) Detect peaks
		min_peak_distance = int(capnobase_fs * 60 / 200)	# 200 BPM
		min_peak_height = 0.8	# 80% of the signal range = avoiding the diastolic peak
		peaks = bc_find_peaks(standardized_window, min_peak_height, min_peak_distance)

		# Add the detected peaks to the list for the corresponding window
		peaks_detected.extend(peaks + start)

	# Remove duplicates and return
	peaks_detected = np.unique(peaks_detected)

	return peaks_detected