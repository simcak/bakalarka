from bcpackage import globals as G
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import correlate

def quality_hjorth():
	from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
	from sklearn.model_selection import train_test_split
	from sklearn.metrics import classification_report, confusion_matrix
	from sklearn.preprocessing import StandardScaler

	# Define sources
	df_capno = pd.read_csv("./hjorth_CBq.csv")
	df_capno["source"] = "capno"

	df_butppg = pd.read_csv("./hjorth_butppg.csv")
	df_butppg["source"] = "but"

	df = pd.concat([
		df_capno,
		df_butppg
	], ignore_index=True)

	# just for but
	df = pd.concat([
		df_butppg
	], ignore_index=True)

	# Define what will we use for classification
	features = ["Mobility Filtered",
				"Complexity Filtered",
				"SPI Filtered",
				"Spectral Ratio",
				"ACF Peaks",
				"Shannon Entropy"
				]
	X = df[features]
	y = (df["Orphanidou Quality"] >= 0.9).astype(int)

	# Scaling for uniformity of Hjorth parameters
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X)

	# Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=42)

	# Model training
	# clf = RandomForestClassifier(n_estimators=100, random_state=42)
	_max_depth = 4
	clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=_max_depth, random_state=42)
	clf.fit(X_train, y_train)

	# Print && show
	y_pred = clf.predict(X_test)
	print("Klasifikační zpráva:")
	print(classification_report(y_test, y_pred, digits=3))
	print("Matice záměn:")
	print(confusion_matrix(y_test, y_pred))

	def _plot_confusion_matrix(y_true, y_pred, class_names=["Špatná kvalita", "Dobrá kvalita"]):
		import seaborn as sns
		from sklearn.metrics import classification_report, confusion_matrix

		# Confusion matrix
		cm = confusion_matrix(y_true, y_pred)
		plt.figure(figsize=(6, 5))
		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
					xticklabels=class_names, yticklabels=class_names)
		plt.xlabel("Predikovaná třída")
		plt.ylabel("Skutečná třída")
		plt.title("Matice záměn (Confusion Matrix)")
		plt.tight_layout()
		plt.show()

		# Classification report
		print("\nKlasifikační zpráva:")
		print(classification_report(y_true, y_pred, target_names=class_names))

	def _plot_feature_importance(clf, feature_names):

	# _plot_confusion_matrix(y_test, y_pred)

		importances = clf.feature_importances_
		indices = np.argsort(importances)[::-1]

		plt.figure(figsize=(10, 6))
		plt.title("Důležitost příznaků (Random Forest)")
		plt.bar(range(len(importances)), importances[indices], align="center")
		plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
		plt.tight_layout()
		plt.grid(True, linestyle='--', alpha=0.5)
		plt.show()

	def _plot_2d_projection(X, y, source, method='pca', title='2D projekce dat'):
		from sklearn.decomposition import PCA
		from sklearn.manifold import TSNE
		import seaborn as sns

		# Dimenzionální redukce
		if method == 'pca':
			X_proj = PCA(n_components=2).fit_transform(X)
		elif method == 'tsne':
			X_proj = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(X)
		else:
			raise ValueError("Podporované metody: 'pca', 'tsne'")

		# Příprava DataFrame pro seaborn
		df_plot = pd.DataFrame({
			"X1": X_proj[:, 0],
			"X2": X_proj[:, 1],
			"Třída": y,
			"Databáze": source
		})

		plt.figure(figsize=(8, 6))
		sns.scatterplot(
			data=df_plot,
			x="X1", y="X2",
			hue="Třída",       # barva = kvalita
			style="Databáze",  # tvar = zdroj dat
			palette=["#d62728", "#1f77b4"],
			alpha=0.7
		)
		plt.title(title)
		plt.xlabel("1. komponenta")
		plt.ylabel("2. komponenta")
		plt.legend(title="Třída / Databáze", loc="best", fontsize=9)
		plt.grid(True, linestyle='--', alpha=0.5)
		plt.tight_layout()
		plt.show()

	_plot_confusion_matrix(y_test, y_pred)
	_plot_feature_importance(clf, features)
	# _plot_2d_projection(X_scaled, y, df["source"], method='pca', title='2D projekce dat (PCA)')
	# _plot_2d_projection(X_scaled, y, df["source"], method='tsne', title='2D projekce dat (t-SNE)')

def hjorth_show_hr(chunked_pieces):
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
	plt.title('Hjorthova TF vs Referenční TF')
	plt.plot(hjorth_hr, label='Hjorthova TF (tepů/min)', marker='o')
	plt.plot(ref_hr, label='Referenční TF (tepů/min)', marker='x')
	plt.xlabel('Index signálu')
	plt.ylabel('Srdeční frekvence (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hjorth_hr.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	# Plot HR difference
	plt.subplot(2, 1, 2)
	plt.title('Rozdíl TF (Hjorthova TF - Referenční TF)')
	plt.plot(hr_diff, label='Rozdíl TF (tepů/min)', color='red', marker='s')
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

def hjorth_show_spi():
	# Plot the relationship between SPI and HR difference
	data = pd.read_csv('./hjorth.csv')
	plt.figure(figsize=(8, 6))
	plt.scatter(data["SPI Filtered"], data["HR diff"], alpha=0.7, c=G.CESA_BLUE, edgecolors='k')

	# Popisky a vzhled
	plt.title("Závislost rozdílu TF na SPI", fontsize=14)
	plt.xlabel("Spectral Purity Index (SPI)", fontsize=12)
	plt.ylabel("Rozdíl TF [bpm]", fontsize=12)
	plt.grid(True)
	plt.tight_layout()

	# Zobraz
	plt.show()

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

def lowpass_filter(signal, cutoff_frequency, sampling_frequency, order=4):
	"""
	Apply a low-pass Butterworth filter to the input signal.
	"""
	from scipy.signal import butter, filtfilt

	nyquist_frequency = 0.5 * sampling_frequency
	normalized_cutoff = cutoff_frequency / nyquist_frequency
	b, a = butter(order, normalized_cutoff, btype='low', analog=False)
	filtered_signal = filtfilt(b, a, signal)

	return filtered_signal

def compute_hjorth_parameters(signal, sampling_frequency, ref_hr, file_id, index, autocorr_iterations,
							  ref_quality=None, only_quality=False, orphanidou_quality=None):
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
	if only_quality and ref_quality == 0:
		return None

	# Name of the file
	if index == 0:
		file_name = file_id
	else:
		file_name = f"{file_id}_{index}min"

	# Compute sampling interval from frequency
	sampling_interval = 1 / sampling_frequency

	# Remove DC offset just in case
	signal -= np.mean(signal)
	# Apply the high-pass filter to remove respiratory frequencies
	respiratory_cutoff_frequency = 0.5  # Example cutoff frequency for respiration [Hz] = equivalent to 30 bpm
	filtered_signal = highpass_filter(signal, respiratory_cutoff_frequency, sampling_frequency)

	# Autocorrelate the signal
	autocorrelated_signal = autocorrelate_signal(filtered_signal, num_iterations=autocorr_iterations)

	############################## HR #############################
	###############################################################
	def _hjorth_hr(signal, sampling_interval):
		# 1st derivative (velocity)
		first_derivative_hr = np.diff(signal)
		# Mean square values
		mean_square_derivative_hr = np.mean(first_derivative_hr ** 2)
		mean_square_signal_hr = np.mean(signal ** 2)
		# Hjorth MOBILITY
		mobility_hr = np.sqrt(mean_square_derivative_hr / mean_square_signal_hr) if mean_square_signal_hr > 0 else 0
		# Convert to Hz (dominant frequency)
		mobility_hz_hr = mobility_hr / (2 * np.pi * sampling_interval)

		return mobility_hz_hr

	mobility_hz_autocorr = _hjorth_hr(autocorrelated_signal, sampling_interval)

	########################### Quality ###########################
	###############################################################
	def _hjorth_quality(signal, fs):
		from scipy.stats import entropy
		from numpy.fft import rfft, rfftfreq

		signal = lowpass_filter(signal, 3.35, fs) # doesnt influence HR but SPI
		# 1st derivative (velocity) and 2nd derivative (acceleration)
		first_derivative_q = np.diff(signal)
		second_derivative_q = np.diff(first_derivative_q)
		# Mean square values
		mean_square_signal_q = np.mean(signal ** 2)
		mean_square_derivative_q = np.mean(first_derivative_q ** 2)
		mean_square_second_derivative_q = np.mean(second_derivative_q ** 2)
		# Safe division to avoid division by zero
		div_1 = mean_square_derivative_q / mean_square_signal_q if mean_square_signal_q > 0 else 0
		div_2 = mean_square_second_derivative_q / mean_square_derivative_q if mean_square_derivative_q > 0 else 0
		# Hjorth parameters
		mobility_q = np.sqrt(div_1) if div_1 > 0 else 0
		complexity_q = np.sqrt(div_2 - div_1) if div_2 > div_1 else 0
		spi_q = np.sqrt(div_1 / div_2) if div_1 > 0 and div_2 > 0 else 0

		# Spectral ratio (0.5–3.5 Hz)
		fft_vals = np.abs(rfft(signal)) ** 2
		freqs = rfftfreq(len(signal), d=1/fs)
		signal_band = (freqs >= 0.5) & (freqs <= 3.5)
		spectral_ratio = fft_vals[signal_band].sum() / fft_vals.sum() if fft_vals.sum() > 0 else 0

		# ACF peak (secondary peak height normalized)
		acf = correlate(signal, signal, mode='full', method='fft')
		acf = acf[acf.size // 2:]  # keep right half
		acf /= np.max(acf) if np.max(acf) > 0 else 1
		acf_peak = np.max(acf[1:])  # skip lag=0

		# Entropy (Shannon, from histogram)
		hist, _ = np.histogram(signal, bins=50, density=True)
		shannon_entropy = entropy(hist + 1e-12)  # small offset to avoid log(0)

		return mobility_q, complexity_q, spi_q, spectral_ratio, acf_peak, shannon_entropy

	mobility_filtr, complexity_filtr, spi_filtr, spectral_ratio, acf_peak, shannon_entropy = _hjorth_quality(filtered_signal, sampling_frequency)
	# mobility_filtr, complexity_filtr, spi_filtr, spectral_ratio, acf_peak, shannon_entropy =None, None, None, None, None, None
	###############################################################

	hjorth_info = {
		"File name": file_name,
		"Domain Freq [Hz]": mobility_hz_autocorr,
		"Hjorth HR": mobility_hz_autocorr * 60,
		"Ref HR": ref_hr,
		"HR diff": abs(ref_hr - (mobility_hz_autocorr * 60)),
		"Mobility Filtered": mobility_filtr,
		"Complexity Filtered": complexity_filtr,
		"SPI Filtered": spi_filtr,
		"Spectral Ratio": spectral_ratio,
		"ACF Peaks": acf_peak,
		"Shannon Entropy": shannon_entropy,
	}
	if orphanidou_quality is not None:
		hjorth_info["Orphanidou Quality"] = orphanidou_quality
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
	if _show and hjorth_info["HR diff"] > 5:
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
		plt.axvline(x=hjorth_info["Domain Freq [Hz]"], color='green', linestyle='--', label=f"Mobilita: {hjorth_info['Domain Freq [Hz]']:.2f} Hz")
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
	from bcpackage import time_count
	from bcpackage.capnopackage import cb_data
	from bcpackage.butpackage import but_data, but_error
	import neurokit2 as nk

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
					fs, file_id = file_info["fs"], file_info["ID"]
					nk_signals, info = nk.ppg_process(chunked_singal, sampling_rate=fs, method_quality="templatematch") # Orphanidou method
					orph_q = np.mean(nk_signals['PPG_Quality'])
					compute_hjorth_parameters(
						chunked_singal, fs, chunk_ref_hr, file_id, j, autocorr_iterations,
						orphanidou_quality=orph_q
						)
			else:
				raise ValueError(f"\033[91m\033[1mInvalid chunk value. Use values in range <1 ; {max_chunk_count}> == <hole signal ; 10s long chunks>\033[0m")

	elif database == "BUT_PPG":
		for i in range(G.BUT_DATA_LEN):
			if chunked_pieces == 1:
				file_info = but_data.extract(i)
				if but_error.police(file_info, i):
					continue
				signal, fs, file_id = file_info['PPG_Signal'], file_info['PPG_fs'], file_info['ID']
				ref_hr, ref_quality = file_info['Ref_HR'], file_info['Ref_Quality']
				nk_signals, info = nk.ppg_process(signal, sampling_rate=fs, method_quality="templatematch") # Orphanidou method
				orph_q = np.mean(nk_signals['PPG_Quality'])
				compute_hjorth_parameters(
					signal, fs, ref_hr, file_id, 0, autocorr_iterations,
					ref_quality=ref_quality, only_quality=False, orphanidou_quality=orph_q
					)
			else:
				raise ValueError("\033[91m\033[1mChunking is not supported for BUT PPG database.\033[0m")
		chunked_pieces = 10

	else:
		raise ValueError("\033[91m\033[1mInvalid database. Use 'CapnoBase' or 'BUT_PPG'.\033[0m")

	# Stop the timer and print the elapsed time
	time_count.stop_terminal_time(start_time, stop_event, func_name=f'Hjorth - {database}')

	# Show the results on graphs and in terminal
	if show:
		hjorth_show_hr(chunked_pieces)
		hjorth_show_spi()
	else:
		print("Hjorth parameters calculated and saved to hjorth.csv.")

	data = pd.read_csv('./hjorth.csv')
	average_hr_diff = data['HR diff'].mean()
	print(f"\033[92m\033[1mPrůměrný rozdíl TF (tepů/min): {average_hr_diff:.2f}\033[0m")
