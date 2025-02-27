from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, export, quality, globals as G

import neurokit2 as nk
import numpy as np
import math

def _compute_global_results(name: str):
	"""
	Function to calculate the global results.
	"""
	total_sensitivity = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FN_LIST))
	total_precision = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FP_LIST))

	export.to_csv_global(f'CB {name} global',
						 total_sensitivity, total_precision,
						 type=name, database='CB')


def capnobase_main_short(method: str, show=False, first=False):
	"""
	Function to run the CapnoBase analysis.
	0. Initialize the lists for the global results with the empty lists.
	1. Load the list of files.
	FOR LOOP:
		2. Iterate over the files.
		3. Extract the CapnoBase data.
		4. Preprocess the PPG signal = filtering = standardization + noise removal.
			- My method
			- NeuroKit method
		5. Detect peaks in the preprocessed signal.
		6. Calculate the heart rate from the detected peaks (using IBI) + calculate the diff with the ref HR.
		7. Confusion matrix: TP, FP, FN.
		8. Performance metrics: Sensitivity, Precision.
		9. Export the data of each signal to a CSV file "./results.csv".

	10. Calculate the global results: Total Sensitivity and Precision + sum of FP, FN, TP + average HR diff.

	Args:
		method (str): The method to execute. Use either "my" or "NeuroKit".
		show (bool): Flag to show debugging or testing info.

	Returns:
		None (exports the results to a CSV file)
	"""
	G.TP_LIST, G.FP_LIST, G.FN_LIST, G.DIFF_HR_LIST, G.QUALITY_LIST = [], [], [], [], []

	for i in range(G.CB_FILES_LEN):
		id, fs, ppg_signal, ref_peaks, hr_info = cb_data.extract(G.CB_FILES[i])

		for chunk_idx in range(8):
			############################################## Chunking ##############################################
			chunk_len = fs * 60
			right_buffer = 0.1 * chunk_len

			right_border_samples = int(chunk_len + right_buffer)

			start_idx = chunk_idx * chunk_len
			end_idx = min(len(ppg_signal), chunk_idx * chunk_len + right_border_samples)

			ppg_chunk = ppg_signal[start_idx:end_idx]

			chunk_ref_peaks = np.array(ref_peaks)
			mask = (chunk_ref_peaks >= start_idx) & (chunk_ref_peaks < end_idx)
			chunk_ref_peaks = chunk_ref_peaks[mask] - start_idx
			######################################################################################################

			if method == 'my':
				filtered_ppg_signal = preprocess.filter_signal(ppg_chunk, fs)
				detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
				quality_info = quality.evaluate(filtered_ppg_signal, detected_peaks,
												fs, None, None,
												method='my_morpho', database='CB')
				name = 'My'

			elif method == 'neurokit':
				nk_signals, nk_info = nk.ppg_process(ppg_chunk, sampling_rate=fs, method="elgendi", method_quality="templatematch")
				detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
				quality_info = quality.evaluate(None, detected_peaks,
												fs, nk_signals['PPG_Quality'], None,
												method='orphanidou', database='CB')
				name = 'NK'
			else:
				raise ValueError(G.INVALID_METHOD)

			local_hr_info = calcul.heart_rate(detected_peaks, hr_info['Ref HR'], fs)
			tp, fp, fn = calcul.confusion_matrix(detected_peaks, chunk_ref_peaks, tolerance=G.TOLERANCE)
			local_sensitivity, local_precision = calcul.performance_metrics(tp, fp, fn)

			export.to_csv_local(id, chunk_idx, i, local_hr_info, quality_info,
								tp, fp, fn, local_sensitivity, local_precision,
								type=name, database='CB', first=first)

			################################### For testing purposes ##################################
			if method == 'my' and show:
				cb_show.test_hub( preprocess.standardize_signal(ppg_chunk), filtered_ppg_signal, chunk_ref_peaks, detected_peaks, local_hr_info, G.CB_FILES[i], i)
			elif method == 'neurokit' and show:
				cb_show.neurokit_show(nk_signals, nk_info, i)

			# Print out chunk-specific results
			if method == 'my':
				print('|  i\t| chunk\t|  ID\t|    Sen\t|    Precision\t|    Diff HR\t|  Our Quality\t|')
			elif method == 'neurokit':
				print('|  i\t| chunk\t|  ID\t|    Sen\t|    Precision\t|    Diff HR\t| Orph. Quality\t|')
			print(f'|  {i}\t|  {chunk_idx}\t|  {id}\t|    {round(local_sensitivity, 3)}\t|    {round(local_precision, 3)}\t|   {round(local_hr_info["Diff HR"], 3)} bpm\t|     {round(quality_info["Calc Q."], 3)}\t|')
			print('----------------------------------------------------------------------------------------')
			############################################################################################

	_compute_global_results(name)
	print('########################################################################################')