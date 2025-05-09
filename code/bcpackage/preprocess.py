from scipy.signal import butter, filtfilt
import numpy as np
from scipy.signal import freqz

import matplotlib.pyplot as plt

def plot_frequency_response(fs, lowcut=0.5, highcut=3.35, order=4):
	"""
	Vykreslí amplitudovou charakteristiku Butterworthova filtru.
	Amplitudová charakteristika je graf, který ukazuje, jak filtr ovlivňuje různé frekvence signálu.

	Parametry:
		fs (float): Vzorkovací frekvence.
		lowcut (float): Dolní mezní frekvence.
		highcut (float): Horní mezní frekvence.
		order (int): Řád filtru.
	"""
	nyquist = 0.5 * fs
	low = lowcut / nyquist
	high = highcut / nyquist
	b, a = butter(order, [low, high], btype="band")
	w, h = freqz(b, a, worN=2000, fs=fs)

	plt.figure(figsize=(13, 5))

	plt.plot(w, np.abs(h), label="Amplitudová odezva", color="black")
	plt.axvspan(lowcut, highcut, color="green", alpha=0.3, label=f"Pásmo průchodu ({lowcut}-{highcut} Hz)")
	plt.title("Amplitudová charakteristika Butterworthova filtru", fontsize=16)
	plt.ylabel("Zesílení [-]", fontsize=14)
	plt.xlabel("Frekvence [Hz]", fontsize=14)
	plt.xlim(0, fs / 2)
	plt.gca().xaxis.set_major_locator(plt.MultipleLocator(5))
	plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
	plt.grid(True, which='both', linestyle='--', linewidth=0.5)
	plt.legend(loc="upper right", fontsize=12)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.tight_layout()

	plt.show()

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

def remove_noise(input_signal, fs, lowcut=0.5, highcut=3.35, order=4):
	"""
	Remove noise from the signal.

	Arguments:
		input_signal and fs are straightforward
		lowcut: lowcut frequency for 30 bpm (0.5 Hz)
		highcut: highcut frequency for 201 bpm (3.35 Hz)
		order: higher order means steeper roll-off, but can cause errors
	"""
	def butter_bandpass(fs, lowcut, highcut, order=4):
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
	# plot_frequency_response(fs, lowcut=0.5, highcut=3.35, order=4)
	denoised_signal = remove_noise(ppg_signal, fs, lowcut=0.5, highcut=3.35)
	# no_BLD_signal = remove_baseline_drift(denoised_signal, fs, highcut=0.5)
	standardized_signal = standardize_signal(denoised_signal)
	# remove motion artifacts?
	output_signal = standardized_signal

	return output_signal