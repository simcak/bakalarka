from . import globals as G
import numpy as np
from scipy.interpolate import interp1d


def compute_beat_features(ppg_signal, peak_index,
						  window_left=0.2,
						  window_right=0.4,
						  fs=100):
	"""
	Extract morphological features for a range around the single beat centered on 'peak_index'.
	
	Parameters
	----------
	ppg_signal : np.ndarray
		1D array containing the cleaned PPG signal.
	peak_index : int
		The index in 'ppg_signal' that corresponds to the peak of interest.
	window_left : float
		Time (seconds) to extend to the left of the peak for analysis (e.g., to find the foot of the wave).
	window_right : float
		Time (seconds) to extend to the right of the peak for analysis (e.g., to see the downslope).
	fs : int
		Sampling frequency of the signal.

	Returns
	-------
	features : dict
		A python dictionary containing morphological metrics (amplitude, rise_time, ...) for one provided beat.
	"""
	########################## PREPROCESSING ##########################
	# Convert windows (in time) to samples. Robust to different fs.
	left_buff_samples = int(window_left * fs)
	right_buff_samples = int(window_right * fs)

	# Define boundaries + we don't go out of the signal range.
	start_idx = max(0, peak_index - left_buff_samples)
	end_idx = min(len(ppg_signal), peak_index + right_buff_samples)

	# EXTRACT THE SEGMENT AROUND THE PEAK
	segment = ppg_signal[start_idx:end_idx]
	segment_peak_idx = peak_index - start_idx

	########################## EXTRACTION ##########################
	systolic_foot_idx_x = np.argmin(segment[:segment_peak_idx + 1])
	systolic_foot_val_y = segment[systolic_foot_idx_x]
	systolic_peak_val_y = segment[segment_peak_idx]

	ppg_amplitude = systolic_peak_val_y - systolic_foot_val_y
	rise_time	  = (segment_peak_idx - systolic_foot_idx_x) / fs #[s]

	########################## STORE FEATURES ##########################
	features = {
		"amplitude": ppg_amplitude,
		"rise_time": rise_time,
		"peak_value_y": systolic_peak_val_y,
		"foot_value_y": systolic_foot_val_y,
		"foot_index_x": systolic_foot_idx_x,
		# placeholder for additional features
	}

	return features


def ppg_quality_morphological(ppg_signal, peaks,
							  fs=100,
							  amplitude_min=0.2,
							  amplitude_max=2.0,
							  rise_time_min=0.04,
							  rise_time_max=0.3):
	"""
	Estimate PPG signal quality based on basic morphological checks:
	- Amplitude threshold (peak - foot).
	- Rise time threshold (time from foot to peak).
	You can add more metrics (e.g., downslope, shape, etc.) to refine quality assessment.

	Parameters
	----------
	ppg_signal : np.ndarray
		The cleaned PPG signal (1D).
	fs : int
		Sampling rate of ppg_signal in Hz.
	amplitude_min : float
		Minimum acceptable amplitude (in the same units as ppg_signal).
	amplitude_max : float
		Maximum acceptable amplitude (in the same units as ppg_signal).
	rise_time_min : float
		Minimum acceptable rise time (seconds).
	rise_time_max : float
		Maximum acceptable rise time (seconds).

	Returns
	-------
	quality_array : np.ndarray
		A continuous 1D array (same length as ppg_signal) where each sample has
		a "quality score" in [0, 1] based on morphological checks.
		- 1.0 indicates "meets all morphological criteria"
		- 0.0 indicates "fails morphological criteria"
		(You can refine this scoring or define intermediate values.)
	morpho_features_per_beat : list of dict
		List of dictionaries, each containing morphological features for each detected beat.
	"""
	beat_quality = np.zeros(len(peaks))	# Init 0 to all beats, then update to 1 if the beat meets the criteria
	morpho_features_per_beat = []

	##################### Evaluate morphological metrics/features for each detected beat #####################
	for i, peak_idx in enumerate(peaks):
		feats = compute_beat_features(ppg_signal, peak_idx, fs=fs)
		morpho_features_per_beat.append(feats)

		# Check if the features meet the criteria
		amplitude_ok = (feats["amplitude"] >= amplitude_min) and (feats["amplitude"] <= amplitude_max)
		rise_time_ok = (feats["rise_time"] >= rise_time_min) and (feats["rise_time"] <= rise_time_max)

		# Basic pass/fail approach (you could combine these in a weighted manner)
		if amplitude_ok and rise_time_ok:
			beat_quality[i] = 1.0
		else:
			beat_quality[i] = 0.0

	##################### Interpolate the quality of each beat to every sample in the signal #####################
	quality_array = np.zeros_like(ppg_signal, dtype=float)

	if len(peaks) > 1:
		peak_position_x = peaks
		beat_quality_y = beat_quality

		# We'll extend the array with a small margin (if no peaks are detected at the beginning/end or bottom/top),
		# so the interpolation covers the entire signal.
		x_interp = np.concatenate(([0], peak_position_x, [len(ppg_signal) - 1]))
		# Undefinied parts of signal inherit the quality of the nearest beat
		y_interp = np.concatenate(([beat_quality_y[0]], beat_quality_y, [beat_quality_y[-1]]))
		# # Undefinied parts of signal are considered as low quality
		# y_interp = np.concatenate(([0], beat_quality_y, [0]))

		# Interpolating itself
		interpolated = interp1d(x_interp, y_interp, kind='linear')  # kind options: 'linear', 'cubic', 'previous'

		# Apply the interpolation to the entire signal (arrange creates vector [0, 1, 2, ..., len(ppg_signal)])
		quality_array = interpolated(np.arange(len(ppg_signal)))

	# less than one peak detected - edge case
	elif len(peaks) == 1:
		quality_array[:] = beat_quality[0]
	else:
		quality_array[:] = 0.0

	return quality_array, morpho_features_per_beat


def evaluate(filtered_ppg_signal, peaks, sampling_rate,
			 quality_arr, ref_quality,
			 method='my_morpho', database='CB'):
	"""
	Calculate the quality of the provided PPG signal.

	Args:
		quality_arr: The quality array of the signal.
		ref_quality [int]: The reference quality (if available).
		method: 'my_morpho' or 'orphanidou'

	Returns:
		avg_quality: The average quality of the signal.
		diff_quality: The difference between the average quality and the reference quality (if provided).
	"""
	if method == 'my_morpho':
		quality_arr, _ = ppg_quality_morphological(filtered_ppg_signal, peaks,
												fs=sampling_rate,
												amplitude_min=G.AMPLITUDE_MIN,
												amplitude_max=G.AMPLITUDE_MAX,
												rise_time_min=G.RISE_TIME_MIN,
												rise_time_max=G.RISE_TIME_MAX)
		# print('Quality array:', quality_arr)
		avg_quality = np.mean(quality_arr)
	elif method == 'orphanidou':
		avg_quality = np.mean(quality_arr)
	else:
		raise ValueError(G.INVALID_QUALITY_METHOD)

	# Round the quality to 0 or 1
	if database == 'BUT':
		if method == 'my_morpho':
			rounded_q = 1 if avg_quality >= G.MORPHO_THRESHOLD else 0
		elif method == 'orphanidou':
			rounded_q = 1 if avg_quality >= G.CORRELATION_THRESHOLD else 0

		# How much is aplied quality alg different from the reference quality?
		diff_quality = abs(rounded_q - ref_quality)
		G.DIFF_QUALITY_SUM += diff_quality

		return avg_quality, diff_quality

	return avg_quality, None