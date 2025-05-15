import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def hjorth_show(chunked_pieces):
	# Load the data from the CSV file
	data = pd.read_csv('./hjorth.csv')

	# Extract relevant columns
	file_names = data['File name']
	hjorth_hr = data['Hjorth HR']
	ref_hr = data['Ref HR']
	hr_diff = data['HR diff']

	# Select every x-th file name for labeling (x = chunked_pieces)
	selected_indices = range(0, len(file_names), chunked_pieces)

	####################### Plot the results #######################
	################################################################
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
	################################################################

def standardize_signal(signal):
	"""
	Standardize the signal to range [-1, 1].
	"""
	min_val = np.min(signal)
	max_val = np.max(signal)
	standardized_signal = 2 * (signal - min_val) / (max_val - min_val) - 1

	return standardized_signal

def autocorrelate_signal(signal, num_iterations=1):
	"""
	Perform repeated autocorrelation on the input signal.
	Autocorrelation is a method to find repeating patterns in the signal.

	Autocorrelation VS Correlation:
	- Autocorrelation: Correlation of a signal with itself at different time lags.
	- Correlation: Correlation of two different signals.
	"""
	from scipy.signal import correlate

	result = signal.copy()
	for _ in range(num_iterations):
		result = correlate(result, result, mode='same', method='fft')	# O(N log N)
		# result = np.correlate(result, result, mode='same')			# O(i * N^2)
	return result

def highpass_filter(signal, cutoff_frequency, sampling_frequency, order=4):
	"""
	Apply a high-pass Butterworth filter to the input signal.
	"""
	from scipy.signal import butter, filtfilt

	nyquist_frequency = 0.5 * sampling_frequency
	normalized_cutoff = cutoff_frequency / nyquist_frequency
	b, a = butter(order, normalized_cutoff, btype='high', analog=False)
	filtered_signal = filtfilt(b, a, signal)
	return filtered_signal

def compute_hjorth_parameters(signal, sampling_frequency, ref_hr, file_id, index, 
							  autocorr_iterations, ref_quality=None, only_quality=False):
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
	if ref_quality == 0 and only_quality:
		return None

	# Name of the file
	if index == 0:
		file_name = file_id
	else:
		file_name = f"{file_id}_{index}min"

	# Compute sampling interval from frequency
	sampling_interval = 1 / sampling_frequency

	# Apply the high-pass filter to remove respiratory frequencies
	respiratory_cutoff_frequency = 0.5  # Example cutoff frequency for respiration [Hz] = equivalent to 30 bpm
	filtered_signal = highpass_filter(signal, respiratory_cutoff_frequency, sampling_frequency)

	# Autocorrelate the signal
	autocorrelated_signal = autocorrelate_signal(filtered_signal, num_iterations=autocorr_iterations)

	############################## HR #############################
	###############################################################
	def _hjorth_hr(signal, sampling_interval):
		# 1st derivative (velocity)
		first_derivative_hr = np.diff(np.insert(signal, 0, 0))
		# Mean square values
		mean_square_derivative_hr = np.mean(first_derivative_hr ** 2)
		mean_square_signal_hr = np.mean(signal ** 2)
		# Hjorth MOBILITY
		mobility_hr = np.sqrt(mean_square_derivative_hr / mean_square_signal_hr)
		# Convert to Hz (dominant frequency)
		mobility_hz_hr = mobility_hr / (2 * np.pi * sampling_interval)

		return mobility_hz_hr

	mobility_hz_autocorr = _hjorth_hr(autocorrelated_signal, sampling_interval)

	########################### Quality ###########################
	###############################################################
	def _hjorth_quality(signal):
		# 1st derivative (velocity) and 2nd derivative (acceleration)
		first_derivative_q = np.diff(np.insert(signal, 0, 0))
		second_derivative_q = np.diff(np.insert(first_derivative_q, 0, 0))
		# Mean square values
		mean_square_signal_q = np.mean(signal ** 2)
		mean_square_derivative_q = np.mean(first_derivative_q ** 2)
		mean_square_second_derivative_q = np.mean(second_derivative_q ** 2)
		# Hjorth MOBILITY
		mobility_q = np.sqrt(mean_square_derivative_q / mean_square_signal_q)
		# Hjorth COMPLEXITY
		complexity_q = np.sqrt((mean_square_second_derivative_q / mean_square_derivative_q) - (mean_square_derivative_q / mean_square_signal_q))
		# Spectral purity index = SPI
		if mean_square_derivative_q > 0 and mean_square_signal_q > 0:
			spi_q = np.sqrt((mean_square_derivative_q / mean_square_signal_q) / (mean_square_second_derivative_q / mean_square_derivative_q))
		else:
			spi_q = 0  # If division by zero occurs

		return mobility_q, complexity_q, spi_q

	mobility_filtr, complexity_filtr, spi_filtr = _hjorth_quality(filtered_signal)
	###############################################################

	hjorth_info = {
		"File name": file_name,
		"Mobility Autocorr [Hz]": mobility_hz_autocorr,
		"Ref HR": ref_hr,
		"Hjorth HR": mobility_hz_autocorr * 60,
		"HR diff": abs(ref_hr - (mobility_hz_autocorr * 60)),
		"Mobility Filtered": mobility_filtr,
		"Complexity Filtered": complexity_filtr,
		"SPI Filtered": spi_filtr,
	}
	if ref_quality is not None:
		hjorth_info["Ref Quality"] = ref_quality

	_export = True
	if _export:
		output_file = "./hjorth.csv"
		try:
			hjorth_df = pd.read_csv(output_file)
		except FileNotFoundError:
			hjorth_df = pd.DataFrame()

		hjorth_df = pd.concat([hjorth_df, pd.DataFrame([hjorth_info])], ignore_index=True)
		hjorth_df.to_csv(output_file, header=True, index=False)

	_show = False
	if _show and hjorth_info["HR diff"] > 42:
		# Frekvenční charakteristika (FFT)
		freqs = np.fft.rfftfreq(len(filtered_signal), d=1/sampling_frequency)
		fft_magnitude = np.abs(np.fft.rfft(filtered_signal))
		freqs_autocorr = np.fft.rfftfreq(len(autocorrelated_signal), d=1/sampling_frequency)
		fft_magnitude_autocorr = np.abs(np.fft.rfft(autocorrelated_signal))

		# Rescale FFT magnitudes to the same scale
		fft_magnitude_rescaled = fft_magnitude / np.max(fft_magnitude)
		fft_magnitude_autocorr_rescaled = fft_magnitude_autocorr / np.max(fft_magnitude_autocorr)

		# Vykreslení autokorelovaného signálu a derivace
		plt.figure(figsize=(12, 6))

		plt.subplot(2, 1, 1)
		plt.title("Přeškálovaný filtrovaný a autokorelovaný signál s derivacemi")
		plt.xlabel("Čas [s]")
		plt.ylabel("Relativní amplituda")
		time_axis = np.arange(len(filtered_signal)) / sampling_frequency
		plt.plot(time_axis, standardize_signal(filtered_signal), label="Filtrovaný signál")
		plt.plot(time_axis, standardize_signal(autocorrelated_signal), label="Autokorelovaný signál")
		plt.legend()
		plt.grid()

		# Vykreslení frekvenční charakteristiky
		plt.subplot(2, 1, 2)
		freq_max = 20
		plt.plot(freqs[freqs <= freq_max], fft_magnitude_rescaled[freqs <= freq_max], label="Frekvenční charakteristika filtrovaného signálu")
		plt.plot(freqs_autocorr[freqs_autocorr <= freq_max], fft_magnitude_autocorr_rescaled[freqs_autocorr <= freq_max], label="Frekvenční charakteristika autokorelovaného signálu")
		plt.axvline(x=hjorth_info["Mobility Autocorr [Hz]"], color='green', linestyle='--', label=f"Mobilita: {hjorth_info['Mobility Autocorr [Hz]']:.2f} Hz")
		plt.title("Přeškálovaná frekvenční charakteristika signálů")
		plt.ylabel("Relativní amplituda")
		plt.xlabel("Frekvence [Hz]")
		plt.legend()
		plt.grid()

		plt.tight_layout()
		plt.show()

	return hjorth_info


def hjorth_alg(database, chunked_pieces=1, autocorr_iterations=4, show=False):
	"""
	Calculate by Hjorth parameters the HR for the given database and evaluate the results.
	"""
	from bcpackage import time_count, globals as G
	from bcpackage.capnopackage import cb_data
	from bcpackage.butpackage import but_data

	def _chunking_signal(chunked_pieces, file_info, chunk_idx):
		"""
		Chunk the signal into smaller segments for processing them.
		Last chunk may be longer than the others.
		Chunking is only supported for CapnoBase database.
		"""
		if chunked_pieces == 1:
			return file_info['Raw Signal'], file_info['Ref HR']

		import numpy as np
		from bcpackage import calcul
		# Calculate the length of each chunk in samples
		chunk_len = len(file_info['Raw Signal']) // chunked_pieces

		start_idx = chunk_idx * chunk_len
		if chunk_idx == chunked_pieces - 1:
			end_idx = len(file_info['Raw Signal'])
		else:
			end_idx = start_idx + chunk_len

		ppg_chunk = file_info['Raw Signal'][start_idx:end_idx]

		chunk_ref_peaks = np.array(file_info['Ref Peaks'])
		mask = (chunk_ref_peaks >= start_idx) & (chunk_ref_peaks < end_idx)
		chunk_ref_peaks = chunk_ref_peaks[mask] - start_idx

		hr_info = calcul.heart_rate(chunk_ref_peaks, None, file_info['fs'], init=True)
		chunk_ref_hr = hr_info['Ref HR']

		return ppg_chunk, chunk_ref_hr

	# Start the timer
	start_time, stop_event = time_count.terminal_time()

	if database == "CapnoBase":
		for i in range(G.CB_FILES_LEN):
			file_info = cb_data.extract(G.CB_FILES[i])
			max_chunk_count = len(file_info['Raw Signal']) // (file_info['fs'] * 10) # 10s long chunks
			# Chunk the signal
			if chunked_pieces >= 1 and chunked_pieces <= max_chunk_count:
				for j in range(chunked_pieces):
					chunked_singal, chunk_ref_hr = _chunking_signal(chunked_pieces, file_info, j)
					compute_hjorth_parameters(chunked_singal, file_info['fs'],
										 chunk_ref_hr, file_info['ID'], j, autocorr_iterations)
			else:
				raise ValueError(f"\033[91m\033[1mInvalid chunk value. Use values in range <1 ; {max_chunk_count}> == <hole signal ; 10s long chunks>\033[0m")

	elif database == "BUT_PPG":
		for i in range(G.BUT_DATA_LEN):
			if chunked_pieces == 1:
				file_info = but_data.extract(i)
				signal, fs, file_id = file_info['PPG_Signal'], file_info['PPG_fs'], file_info['ID']
				ref_hr, ref_quality = file_info['Ref_HR'], file_info['Ref_Quality']
				compute_hjorth_parameters(signal, fs, ref_hr, file_id, 0, autocorr_iterations,
									 ref_quality=ref_quality, only_quality=True)
			else:
				raise ValueError("\033[91m\033[1mChunking is not supported for BUT PPG database.\033[0m")
		chunked_pieces = 10

	else:
		raise ValueError("\033[91m\033[1mInvalid database. Use 'CapnoBase' or 'BUT_PPG'.\033[0m")

	# Stop the timer and print the elapsed time
	time_count.stop_terminal_time(start_time, stop_event, func_name=f'Hjorth - {database}')

	# Show the results on graphs and in terminal
	if show:
		hjorth_show(chunked_pieces)
	else:
		print("Hjorth parameters calculated and saved to hjorth.csv.")

	data = pd.read_csv('./hjorth.csv')
	average_hr_diff = data['HR diff'].mean()
	print(f"\033[92m\033[1mPrůměrný rozdíl TF (tepů/min): {average_hr_diff:.2f}\033[0m")
