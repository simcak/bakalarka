from scipy.signal import butter, filtfilt

def remove_noise(input_signal, fs, lowcut=0.5, highcut=4.0, order=4):
	"""
	Remove noise from the signal.

	Arguments:
		input_signal and fs are straightforward
		lowcut: lowcut frequency for 30 bpm
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
def filter_signal(ppg_signal, capnobase_fs):
	"""
	Filter the PPG signal.
	"""
	denoised_signal = remove_noise(ppg_signal, capnobase_fs, lowcut=0.5, highcut=4.0)
	# no_BLD_signal = remove_baseline_drift(denoised_signal, capnobase_fs, highcut=0.5)
	standardized_signal = (denoised_signal - denoised_signal.mean()) / denoised_signal.std()
	# remove motion artifacts?
	output_signal = standardized_signal

	return output_signal