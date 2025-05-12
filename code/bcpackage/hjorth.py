import numpy as np
import matplotlib.pyplot as plt
from bcpackage import calcul
import csv

def standardize_signal(signal):
	"""
	Standardize the signal to range [-1, 1].

	Parameters:
	- signal: numpy array, input signal

	Returns:
	- standardized_signal: numpy array, signal scaled to [-1, 1]
	"""
	min_val = np.min(signal)
	max_val = np.max(signal)
	standardized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1

	return standardized_signal

def autocorrelate_signal(signal, num_iterations=1):
	"""
	Perform repeated autocorrelation on the input signal.

	Parameters:
	- signal: numpy array, raw input signal
	- num_iterations: int, number of times to apply autocorrelation

	Returns:
	- autocorrelated_signal: numpy array, result after repeated autocorrelation
	"""
	result = signal.copy()
	for _ in range(num_iterations):
		result = np.correlate(result, result, mode='same')
		# Normalize the result to avoid scaling issues
		result = result / np.max(result)
	return result

def compute_hjorth_parameters(signal, sampling_frequency, detected_peaks, ref_hr, capnobase_file, index=0, quality=None):
	"""
	Compute Hjorth parameters: mobility, complexity, and spectral purity index (SPI).

	Parameters:
	- signal: numpy array (n x 1), scalar time series
	- sampling_frequency: float, sampling frequency [Hz]

	Returns:
	- mobility_hz: dominant frequency (mobility / (2 * pi * T)) [Hz]
	- complexity_hz: bandwidth (complexity / (2 * pi * T)) [Hz]
	- spectral_purity_index: SPI, value between 0 and 1, 1 means pure harmonic
	"""
	if quality == 0:
		return None

	# Name of the file
	if index == 0:
		file_name = capnobase_file
	else:
		file_name = f"{capnobase_file}_{index}min"

	# Compute sampling interval from frequency
	sampling_interval = 1 / sampling_frequency

	# DC removal
	signal = signal - np.mean(signal)

	# Respiration removal - high pass
	def highpass_filter(signal, cutoff_frequency, sampling_frequency, order=4):
		from scipy.signal import butter, filtfilt

		nyquist_frequency = 0.5 * sampling_frequency
		normalized_cutoff = cutoff_frequency / nyquist_frequency
		b, a = butter(order, normalized_cutoff, btype='high', analog=False)
		filtered_signal = filtfilt(b, a, signal)
		return filtered_signal

	# Apply the high-pass filter to remove respiratory frequencies
	respiratory_cutoff_frequency = 0.5  # Example cutoff frequency for respiration [Hz] = equivalent to 30 bpm
	signal = highpass_filter(signal, respiratory_cutoff_frequency, sampling_frequency)

	# Autocorrelate the signal
	autocorrelated_signal = autocorrelate_signal(signal, num_iterations=3)

	# First derivative (velocity)
	first_derivative = np.diff(np.insert(autocorrelated_signal, 0, 0))
	# Second derivative (acceleration)
	second_derivative = np.diff(np.insert(first_derivative, 0, 0))

	# Mean square values
	mean_square_signal = np.mean(autocorrelated_signal ** 2)
	mean_square_derivative = np.mean(first_derivative ** 2)
	mean_square_second_derivative = np.mean(second_derivative ** 2)

	# Hjorth MOBILITY
	mobility = np.sqrt(mean_square_derivative / mean_square_signal)
	# Hjorth COMPLEXITY
	complexity = np.sqrt((mean_square_second_derivative / mean_square_derivative) - (mean_square_derivative / mean_square_signal))
	# Spectral purity index = SPI
	spectral_purity_index = np.sqrt((mean_square_derivative / mean_square_signal) / (mean_square_second_derivative / mean_square_derivative))

	# Convert to Hz (dominant frequency and bandwidth)
	mobility_hz = mobility / (2 * np.pi * sampling_interval)
	complexity_hz = complexity / (2 * np.pi * sampling_interval)

	hjorth_info = {
		"File name": file_name,
		"Mobility": mobility_hz,
		"Complexity": complexity_hz,
		"Spectral Purity Index": spectral_purity_index,
	}
	if quality is not None:
		hjorth_info["Ref Quality"] = quality
	# Calculate the heart rate
	local_hr_info = calcul.heart_rate(detected_peaks, ref_hr, sampling_frequency)
	hjorth_info["Hjorth HR"] = mobility_hz * 60
	hjorth_info["Ref HR"] = local_hr_info['Calculated HR']
	hjorth_info["HR diff"] = abs(local_hr_info['Calculated HR'] - (hjorth_info['Hjorth HR']))

	_export = True
	if _export:
		import pandas as pd

		output_file = "./hjorth.csv"
		try:
			hjorth_df = pd.read_csv(output_file)
		except FileNotFoundError:
			hjorth_df = pd.DataFrame()

		hjorth_df = pd.concat([hjorth_df, pd.DataFrame([hjorth_info])], ignore_index=True)
		hjorth_df.to_csv(output_file, header=True, index=False)

	show = False
	if show and hjorth_info["HR diff"] > 42:
		# Frekvenční charakteristika (FFT)
		freqs = np.fft.rfftfreq(len(signal), d=1/sampling_frequency)
		fft_magnitude = np.abs(np.fft.rfft(signal))
		freqs_autocorr = np.fft.rfftfreq(len(autocorrelated_signal), d=1/sampling_frequency)
		fft_magnitude_autocorr = np.abs(np.fft.rfft(autocorrelated_signal))

		# Vykreslení autokorelovaného signálu a derivace
		plt.figure(figsize=(12, 6))

		plt.subplot(2, 1, 1)
		plt.title("Původní a autokorelovaný signál s derivacemi")
		plt.xlabel("Čas [s]")
		plt.ylabel("Amplituda")
		time_axis = np.arange(len(signal)) / sampling_frequency
		plt.plot(time_axis, standardize_signal(signal), label="Původní signál")
		plt.plot(time_axis, standardize_signal(autocorrelated_signal), label="Autokorelovaný signál")
		plt.legend()
		plt.grid()

		# Vykreslení frekvenční charakteristiky
		plt.subplot(2, 1, 2)
		plt.plot(freqs, fft_magnitude, label="Frekvenční charakteristika původního signálu")
		plt.plot(freqs_autocorr, fft_magnitude_autocorr, label="Frekvenční charakteristika autokorelovaného signálu")
		plt.title("Frekvenční charakteristika signálů")
		plt.xlabel("Frekvence [Hz]")
		plt.ylabel("Amplituda")
		plt.legend()
		plt.grid()

		plt.tight_layout()
		plt.show()

	_print = False
	if _print:
		local_hr_info = calcul.heart_rate(detected_peaks, ref_hr, sampling_frequency)
		print("************ Hjorth Parameters:")
		print(f"Mobility (Hz):           {hjorth_info['Mobility']:.2f}")
		print(f"Hjorth HR (bpm):         {hjorth_info['Hjorth HR']:.2f}")
		print(f"Our HR (bpm):            {local_hr_info['Calculated HR']:.2f}")
		print(f"\033[91mHR Difference (bpm):     {abs(local_hr_info['Calculated HR'] - (hjorth_info['Hjorth HR'])):.2f}\033[0m")
		print(f"Complexity (Hz):         {hjorth_info['Complexity']:.2f}")
		print(f"Spectral Purity Index:   {hjorth_info['Spectral Purity Index']:.2f}\n")

	return hjorth_info

def hjorth_final_calculation():
	import pandas as pd
	# Load the data from the CSV file
	data = pd.read_csv('./hjorth.csv')

	# Extract relevant columns
	file_names = data['File name']
	hjorth_hr = data['Hjorth HR']
	ref_hr = data['Ref HR']
	hr_diff = data['HR diff']

	# Select every 16th file name for labeling
	selected_indices = range(0, len(file_names), 16)
	selected_file_names = file_names.iloc[selected_indices]

	# Plot the results
	plt.figure(figsize=(12, 6))

	# Plot Hjorth HR and Ref HR
	plt.subplot(2, 1, 1)
	plt.plot(hjorth_hr, label='Hjorthova TF (tepů/min)', marker='o')
	plt.plot(ref_hr, label='Referenční TF (tepů/min)', marker='x')

	plt.title('Hjorthova TF vs Referenční TF')
	plt.xlabel('Index signálu')
	plt.ylabel('Srdeční frekvence (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hjorth_hr.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	# Plot HR difference
	plt.subplot(2, 1, 2)
	plt.plot(hr_diff, label='Rozdíl TF (tepů/min)', color='red', marker='s')
	plt.title('Rozdíl TF (Hjorthova TF - Referenční TF)')
	plt.xlabel('Index signálu')
	plt.ylabel('Rozdíl TF (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hr_diff.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	plt.tight_layout()
	plt.show()

	# Calculate and print the average HR difference
	average_hr_diff = hr_diff.mean()
	print(f"Průměrný rozdíl TF (tepů/min): {average_hr_diff:.2f}")
