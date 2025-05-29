def empty():
	# from . import globals as G
	# import numpy as np
	# from scipy.interpolate import interp1d


	# def compute_beat_features(ppg_signal, peak_index,
	# 						  border_left=0.2,
	# 						  border_right=0.4,
	# 						  fs=100):
	# 	"""
	# 	Extract morphological features for a range around the single beat centered on 'peak_index'.
		
	# 	Parameters
	# 	----------
	# 	ppg_signal : np.ndarray
	# 		1D array containing the cleaned PPG signal.
	# 	peak_index : int
	# 		The index in 'ppg_signal' that corresponds to the peak of interest.
	# 	border_left : float
	# 		Time (seconds) to extend to the left of the peak for analysis (e.g., to find the foot of the wave).
	# 	border_right : float
	# 		Time (seconds) to extend to the right of the peak for analysis (e.g., to see the downslope).
	# 	fs : int
	# 		Sampling frequency of the signal.

	# 	Returns
	# 	-------
	# 	features : dict
	# 		A python dictionary containing morphological metrics (amplitude, rise_time, ...) for one provided beat.
	# 	"""
	# 	########################## PREPROCESSING ##########################
	# 	# Convert borders (in time) to samples. Robust for different fs.
	# 	left_border_samples = int(border_left * fs)
	# 	right_border_samples = int(border_right * fs)

	# 	# Define boundaries + we don't go out of the signal range.
	# 	start_idx = max(0, peak_index - left_border_samples)
	# 	end_idx = min(len(ppg_signal), peak_index + right_border_samples)

	# 	# EXTRACT THE SEGMENT AROUND THE PEAK
	# 	segment = ppg_signal[start_idx:end_idx]
	# 	segment_peak_idx = peak_index - start_idx

	# 	########################## EXTRACTION ##########################
	# 	systolic_foot_idx_x = np.argmin(segment[:segment_peak_idx + 1])
	# 	systolic_foot_val_y = segment[systolic_foot_idx_x]
	# 	systolic_peak_val_y = segment[segment_peak_idx]

	# 	ppg_amplitude = systolic_peak_val_y - systolic_foot_val_y
	# 	rise_time	  = (segment_peak_idx - systolic_foot_idx_x) / fs #[s]

	# 	########################## STORE FEATURES ##########################
	# 	features = {
	# 		"amplitude": ppg_amplitude,
	# 		"rise_time": rise_time,
	# 		"peak_value_y": systolic_peak_val_y,
	# 		"foot_value_y": systolic_foot_val_y,
	# 		"foot_index_x": systolic_foot_idx_x,
	# 		# placeholder for additional features
	# 	}

	# 	return features


	# def ppg_quality_morphological(ppg_signal, peaks,
	# 							  fs=100,
	# 							  amplitude_min=0.2,
	# 							  amplitude_max=2.0,
	# 							  rise_time_min=0.04,
	# 							  rise_time_max=0.3):
	# 	"""
	# 	Estimate PPG signal quality based on basic morphological checks:
	# 	- Amplitude threshold (peak - foot).
	# 	- Rise time threshold (time from foot to peak).
	# 	You can add more metrics (e.g., downslope, shape, etc.) to refine quality assessment.

	# 	Parameters
	# 	----------
	# 	ppg_signal : np.ndarray
	# 		The cleaned PPG signal (1D).
	# 	fs : int
	# 		Sampling rate of ppg_signal in Hz.
	# 	amplitude_min : float
	# 		Minimum acceptable amplitude (in the same units as ppg_signal).
	# 	amplitude_max : float
	# 		Maximum acceptable amplitude (in the same units as ppg_signal).
	# 	rise_time_min : float
	# 		Minimum acceptable rise time (seconds).
	# 	rise_time_max : float
	# 		Maximum acceptable rise time (seconds).

	# 	Returns
	# 	-------
	# 	quality_array : np.ndarray
	# 		A continuous 1D array (same length as ppg_signal) where each sample has
	# 		a "quality score" in [0, 1] based on morphological checks.
	# 		- 1.0 indicates "meets all morphological criteria"
	# 		- 0.0 indicates "fails morphological criteria"
	# 		(You can refine this scoring or define intermediate values.)
	# 	morpho_features_per_beat : list of dict
	# 		List of dictionaries, each containing morphological features for each detected beat.
	# 	"""
	# 	beat_quality = np.zeros(len(peaks))	# Init 0 to all beats, then update to 1 if the beat meets the criteria
	# 	morpho_features_per_beat = []

	# 	##################### Evaluate morphological metrics/features for each detected beat #####################
	# 	for i, peak_idx in enumerate(peaks):
	# 		feats = compute_beat_features(ppg_signal, peak_idx, fs=fs)
	# 		morpho_features_per_beat.append(feats)

	# 		# Check if the features meet the criteria
	# 		amplitude_ok = (feats["amplitude"] >= amplitude_min) and (feats["amplitude"] <= amplitude_max)
	# 		rise_time_ok = (feats["rise_time"] >= rise_time_min) and (feats["rise_time"] <= rise_time_max)

	# 		# Basic pass/fail approach (you could combine these in a weighted manner)
	# 		if amplitude_ok and rise_time_ok:
	# 			beat_quality[i] = 1.0
	# 		else:
	# 			beat_quality[i] = 0.0

	# 	##################### Interpolate the quality of each beat to every sample in the signal #####################
	# 	quality_array = np.zeros_like(ppg_signal, dtype=float)

	# 	if len(peaks) > 1:
	# 		peak_position_x = peaks
	# 		beat_quality_y = beat_quality

	# 		# We'll extend the array with a small margin (if no peaks are detected at the beginning/end or bottom/top),
	# 		# so the interpolation covers the entire signal.
	# 		x_interp = np.concatenate(([0], peak_position_x, [len(ppg_signal) - 1]))
	# 		# Undefinied parts of signal inherit the quality of the nearest beat
	# 		y_interp = np.concatenate(([beat_quality_y[0]], beat_quality_y, [beat_quality_y[-1]]))
	# 		# # Undefinied parts of signal are considered as low quality
	# 		# y_interp = np.concatenate(([0], beat_quality_y, [0]))

	# 		# Interpolating itself
	# 		interpolated = interp1d(x_interp, y_interp, kind='linear')  # kind options: 'linear', 'cubic', 'previous'

	# 		# Apply the interpolation to the entire signal (arrange creates vector [0, 1, 2, ..., len(ppg_signal)])
	# 		quality_array = interpolated(np.arange(len(ppg_signal)))

	# 	# less than one peak detected - edge case
	# 	elif len(peaks) == 1:
	# 		quality_array[:] = beat_quality[0]
	# 	else:
	# 		quality_array[:] = 0.0

	# 	return quality_array, morpho_features_per_beat


	# def evaluate(filtered_ppg_signal, peaks, sampling_rate,
	# 			 quality_arr, ref_quality,
	# 			 method='my_morpho', database='CB'):
	# 	"""
	# 	Calculate the quality of the provided PPG signal.

	# 	Args:
	# 		quality_arr: The quality array of the signal.
	# 		ref_quality [int]: The reference quality (if available).
	# 		method: 'my_morpho' or 'orphanidou'

	# 	Returns:
	# 		quality_info: A dictionary containing quality information.
	# 	"""
	# 	if method == 'my_morpho':
	# 		quality_arr, morpho_data = ppg_quality_morphological(filtered_ppg_signal, peaks,
	# 												fs=sampling_rate,
	# 												amplitude_min=G.AMPLITUDE_MIN,
	# 												amplitude_max=G.AMPLITUDE_MAX,
	# 												rise_time_min=G.RISE_TIME_MIN,
	# 												rise_time_max=G.RISE_TIME_MAX)
	# 		# print('Quality array:', quality_arr)
	# 		quality_out = np.mean(quality_arr)
	# 		G.QUALITY_LIST.append(quality_out)

	# 	elif method == 'orphanidou':
	# 		quality_out = np.mean(quality_arr)
	# 		G.QUALITY_LIST.append(quality_out)

	# 	else:
	# 		raise ValueError(G.INVALID_QUALITY_METHOD)

	# 	# Round the quality to 0 or 1
	# 	if database == 'BUT':
	# 		if method == 'my_morpho':
	# 			rounded_q = 1 if quality_out >= G.MORPHO_THRESHOLD else 0
	# 		elif method == 'orphanidou':
	# 			rounded_q = 1 if quality_out >= G.CORRELATION_THRESHOLD else 0
	# 		# How much is aplied quality alg different from the reference quality?
	# 		diff_quality = abs(rounded_q - ref_quality)
	# 		G.DIFF_QUALITY_SUM += diff_quality

	# 	elif database == 'CB':
	# 		rounded_q = None
	# 		diff_quality = None

	# 	quality_info = {
	# 		'Q. array': quality_arr,
	# 		'Calc Q.': quality_out,
	# 		'Rounded Q.': rounded_q,
	# 		'Diff Q.': diff_quality,
	# 		'Ref Q.': None
	# 	}

	# 	return quality_info
	pass

import numpy as np
import pandas as pd

def orphanidou_quality_plot(threshold, chunked_pieces):
	import matplotlib.pyplot as plt
	from bcpackage import globals as G

	# Load the data
	orph_data = pd.read_csv("./orph_q_but.csv")

	# Extract values
	signal_ids = orph_data["ID"].astype(str).values
	x_idx = np.arange(len(signal_ids))
	orph_quality = orph_data['Orphanidou_Quality'].values
	ref_quality = orph_data['Ref_Quality'].values

	# chunked_pieces = 1
	# signal_ids = signal_ids[:48]
	# x_idx = x_idx[:48]
	# orph_quality = orph_quality[:48]
	# ref_quality = ref_quality[:48]

	# Determine colors based on equality
	colors = [G.CESA_BLUE if oq >= threshold and rq == 1 else G.BUT_RED for oq, rq in zip(orph_quality, ref_quality)]

	plt.figure(figsize=(10, 13))
	plt.scatter(x_idx, orph_quality, c=colors, alpha=0.7, edgecolors='k', s=20)
	# plt.axhline(y=threshold, color="grey", linestyle="--", linewidth=1)
	plt.title("Orphanidou vs Referenční kvalita BUT PPG", fontsize=16)
	plt.ylabel("Orphanidou kvalita", fontsize=14)
	plt.xlabel("ID signálu", fontsize=14)
	tick_positions = x_idx[::chunked_pieces]
	tick_labels = signal_ids[::chunked_pieces]
	tick_labels_mod = []
	for i, label in enumerate(tick_labels):
		if i % 2 == 0:
			tick_labels_mod.append(label)
		else:
			tick_labels_mod.append(' ')
	plt.xticks(tick_positions, tick_labels_mod, rotation=90)
	plt.grid(axis="y", alpha=0.3)
	plt.tight_layout()
	plt.legend(
		handles=[
			plt.Line2D([0], [0], marker='o', color='w', label='Kvalitní dle databáze', markerfacecolor=G.CESA_BLUE, markersize=10),
			plt.Line2D([0], [0], marker='o', color='w', label='Negativní dle databáze', markerfacecolor=G.BUT_RED, markersize=10),
			# plt.Line2D([0], [0], color='grey', linestyle="--", linewidth=1, label=f'Threshold: {threshold}'),
		],
		loc='upper right',
		fontsize=13,
	)
	plt.show()

def orphanidou_quality_evaluation(threshold, print_out=False):
	orph_data = pd.read_csv("./orphanidou_quality.csv")

	signal_id = orph_data['ID'].values
	orph_quality = (orph_data['Orphanidou_Quality'].values >= threshold).astype(int)
	ref_quality = orph_data['Ref_Quality'].values

	# Confusion matrix
	TP = np.sum((orph_quality == 1) & (ref_quality == 1))
	FP = np.sum((orph_quality == 1) & (ref_quality == 0))
	TN = np.sum((orph_quality == 0) & (ref_quality == 0))
	FN = np.sum((orph_quality == 0) & (ref_quality == 1))

	Se = TP / (TP + FN) if (TP + FN) > 0 else 0		# sensitivity / recall
	PPV = TP / (TP + FP) if (TP + FP) > 0 else 0	# positive predictive value / precision
	F1 = 2 * (PPV * Se) / (PPV + Se) if (PPV + Se) > 0 else 0

	if print_out:
		# print and show the success of Orphanidou quality
		print("---------------------------------------------------------------------------")
		print("Confusion Matrix:")
		print(f"\t\t\tActually TRUE\tActually FALSE")
		print(f"Predicted POSITIVE\t{TP}\t\t{FP}")
		print(f"Predicted NEGATIVE\t{FN}\t\t{TN}")
		print("---------------------------------------------------------------------------")
		print(f"Se: {Se:.3f}, PPV: {PPV:.3f}, F1: {F1:.3f}")

	return F1

def ref_quality_orphanidou(database='BUT_PPG', chunked_pieces=1):
	from bcpackage import globals as G, time_count
	from bcpackage.butpackage import but_data, but_error
	from bcpackage.capnopackage import cb_data
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

	def _export_to_csv(_orphanidou_info):
		output_file = "./orphanidou_quality.csv"
		try:
			orphanidou_df = pd.read_csv(output_file)
		except FileNotFoundError:
			orphanidou_df = pd.DataFrame()
		orphanidou_df = pd.concat([orphanidou_df, pd.DataFrame([_orphanidou_info])], ignore_index=True)
		orphanidou_df.to_csv(output_file, header=True, index=False)

	start_time, stop_event = time_count.terminal_time()

	if database == 'CapnoBase':
		for i in range(G.CB_FILES_LEN):
			file_info = cb_data.extract(G.CB_FILES[i])
			max_chunk_count = len(file_info['Raw Signal']) // (file_info['fs'] * 10) # 10s long chunks
			# Chunk the signal
			if chunked_pieces >= 1 and chunked_pieces <= max_chunk_count:
				for j in range(chunked_pieces):
					signal, ref_hr = _chunking_signal(chunked_pieces, file_info, j)
					fs, file_id, ref_quality = file_info['fs'], file_info['ID'], None
					nk_signals, info = nk.ppg_process(signal, sampling_rate=fs, method_quality="templatematch") # Orphanidou method
					orphanidou_quality = np.mean(nk_signals['PPG_Quality'])
					file_id = file_id if j == 0 else f"{file_id}_{j+1}"

					orphanidou_info = {
						'ID': file_id,
						'fs': fs,
						'Ref_HR': ref_hr,
						'Ref_Quality': ref_quality,
						'Orphanidou_Quality': orphanidou_quality,
					}
					_export_to_csv(orphanidou_info)
			else:
				raise ValueError(f"Chunked pieces must be between 1 and {max_chunk_count}.")

	elif database == 'BUT_PPG':
		for i in range(G.BUT_DATA_LEN):
			file_info = but_data.extract(i)
			if but_error.police(file_info, i):
				continue
			signal, fs, file_id = file_info['PPG_Signal'], file_info['PPG_fs'], file_info['ID']
			ref_hr, ref_quality = file_info['Ref_HR'], file_info['Ref_Quality']
			nk_signals, info = nk.ppg_process(signal, sampling_rate=fs, method_quality="templatematch") # Orphanidou method
			orphanidou_quality = np.mean(nk_signals['PPG_Quality'])

			orphanidou_info = {
				'ID': file_id,
				'fs': fs,
				'Ref_HR': ref_hr,
				'Ref_Quality': ref_quality,
				'Orphanidou_Quality': orphanidou_quality,
			}
			_export_to_csv(orphanidou_info)

	time_count.stop_terminal_time(start_time, stop_event, func_name=f'Orphanidou quality')

