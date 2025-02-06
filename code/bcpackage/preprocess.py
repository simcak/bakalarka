from scipy.signal import butter, filtfilt
import numpy as np

def standardize_signal(signal):
	"""
	Standardizes a signal to the range [-1, 1].

	Parameters:
		signal (array-like): The input signal.

	Returns:
		standardized_signal (array-like): The standardized signal with values between -1 and 1.
	"""
	# Center the signal by subtracting the mean
	centered_signal = signal - np.mean(signal)
	
	# Normalize by the standard deviation
	standardized_signal = centered_signal / np.std(centered_signal)
	
	# Scale to the range [-1, 1]
	min_val = np.min(standardized_signal)
	max_val = np.max(standardized_signal)
	standardized_signal = 2 * (standardized_signal - min_val) / (max_val - min_val) - 1
	
	return standardized_signal

def remove_noise(input_signal, fs, lowcut=0.7, highcut=4.0, order=4):
	"""
	Remove noise from the signal.

	Arguments:
		input_signal and fs are straightforward
		lowcut: lowcut frequency for 42 bpm
		highcut: highcut frequency for 200 bpm
		order: higher order means steeper roll-off, but can cause errors
	"""
	def butter_bandpass(fs, lowcut, highcut, order=5):
		"""
		Butterworth bandpass filter.
		"""
		nyquist = 0.5 * fs
		low = lowcut / nyquist
		high = highcut / nyquist
		b, a = butter(order, [low, high], btype='band')
		return b, a

	b, a = butter_bandpass(fs, lowcut, highcut, order=order)
	output_signal = filtfilt(b, a, input_signal)

	return output_signal

# Filter functions crossroad
def filter_signal(ppg_signal, fs):
	"""
	Filter the PPG signal.
	"""
	denoised_signal = remove_noise(ppg_signal, fs, lowcut=0.7, highcut=4.0)
	# no_BLD_signal = remove_baseline_drift(denoised_signal, fs, highcut=0.5)
	standardized_signal = standardize_signal(denoised_signal)
	# remove motion artifacts?
	output_signal = standardized_signal

	return output_signal