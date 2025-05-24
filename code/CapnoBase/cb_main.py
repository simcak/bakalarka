from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, export, globals as G, time_count, hjorth

import neurokit2 as nk
import numpy as np

def _compute_global_results(name: str):
	"""
	Function to calculate the global results.

	Args:
		name (str): The name of the method.
	"""
	total_sensitivity = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FN_LIST))
	total_precision = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FP_LIST))

	export.to_csv_global(total_sensitivity, total_precision, type=name, database='CB')

def _chunking_signal(file_info, chunk_idx):
	"""
	Chunk the signal.

	Args:
		file_info (dict): The file info.
		chunk_idx (int): The index of the chunk.
	"""
	chunk_len = file_info['fs'] * 60
	right_buffer = chunk_len * 0.05 # for 5% overlap

	right_border_samples = int(chunk_len + right_buffer)

	start_idx = chunk_idx * chunk_len
	end_idx = min(len(file_info['Raw Signal']), chunk_idx * chunk_len + right_border_samples)

	ppg_chunk = file_info['Raw Signal'][start_idx:end_idx]

	chunk_ref_peaks = np.array(file_info['Ref Peaks'])
	mask = (chunk_ref_peaks >= start_idx) & (chunk_ref_peaks < end_idx)
	chunk_ref_peaks = chunk_ref_peaks[mask] - start_idx

	hr_info = calcul.heart_rate(chunk_ref_peaks, None, file_info['fs'], init=True)
	chunk_ref_hr = hr_info['Ref HR']

	return ppg_chunk, chunk_ref_peaks, chunk_ref_hr

def _process_signal(file_info, method, i, show, chunk=False, chunk_idx=0):
	"""
	Process the signal according to the method.

	Args:
		file_info (dict): The info comming from the file.
		method (str): The method to execute.
		i (int): The index of the file.
		show (bool): Flag to show signals - for testing purposes.
		chunk (bool): Flag to chunk the signal.
		chunk_idx (int): The index of the chunk.
	"""
	# init & assign
	nk_signals, nk_info, filtered_ppg_signal = None, None, None
	fs = file_info['fs']

	if chunk:
		ppg_signal, ref_peaks, ref_hr = _chunking_signal(file_info, chunk_idx)
	else:
		ppg_signal, ref_peaks, ref_hr = file_info['Raw Signal'], file_info['Ref Peaks'], file_info['Ref HR']

	# Execute my method
	if method == 'my':
		filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
		detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
		name = 'My'

	# Execute the NeuroKit package with:
	#	Elgendi method for peak detection and
	#	Orphanidou method for quality estimation == templatematch
	#	nk_signal returns: PPG_Raw  PPG_Clean  PPG_Rate  PPG_Quality  PPG_Peaks for each sample
	elif method == 'neurokit':
		nk_signals, nk_info = nk.ppg_process(ppg_signal, sampling_rate=fs,
									   method="elgendi", method_quality="templatematch")
		detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
		name = 'NK'

	else:
		raise ValueError(G.INVALID_METHOD)

	local_hr_info = calcul.heart_rate(detected_peaks, ref_hr, fs)
	tp, fp, fn = calcul.confusion_matrix(detected_peaks, ref_peaks, tolerance=G.TOLERANCE)
	local_sensitivity, local_precision = calcul.performance_metrics(tp, fp, fn)

	statistical_info = {
		'TP': tp, 'FP': fp, 'FN': fn,
		'Se': local_sensitivity, 'PPV': local_precision
	}

	################################### For testing purposes ##################################
	if method == 'my' and show:
		cb_show.test_hub(preprocess.standardize_normalize_signal(ppg_signal), filtered_ppg_signal, ref_peaks, detected_peaks, local_hr_info, G.CB_FILES[i], i)
	elif method == 'neurokit' and show:
		cb_show.neurokit_show(nk_signals, nk_info, i)
	############################################################################################

	return name, local_hr_info, statistical_info

def capnobase_main(method: str, chunk=False, show=False, first=False):
	"""
	Function to run the CapnoBase analysis.

	Args:
		method (str): The method to execute. Use either "my" or "NeuroKit".
		chunk (bool): Flag to chunk the signal.
		show (bool): Flag to show debugging or testing info.
		first (bool): Flag to indicate the first run for the CSV file.

	Returns:
		None (exports the results to a CSV file)
	"""
	G.TP_LIST, G.FP_LIST, G.FN_LIST, G.DIFF_HR_LIST = [], [], [], []
	start_time, stop_event = time_count.terminal_time()

	for i in range(G.CB_FILES_LEN):
		file_info = cb_data.extract(G.CB_FILES[i])

		if chunk:
			for chunk_idx in range(8):
				name, local_hr_info, statistical_info = _process_signal(file_info, method, i, show,
																			chunk=chunk, chunk_idx=chunk_idx)

				export.to_csv_local(file_info['ID'], chunk_idx, i, local_hr_info, statistical_info,
							  type=name, database='CB', first=first)
		else:
			name, local_hr_info, statistical_info = _process_signal(file_info, method, i, show)

			export.to_csv_local(file_info['ID'], 8, i, local_hr_info, statistical_info,
						  type=name, database='CB', first=first)

	_compute_global_results(name)
	if chunk == True:
		time_count.stop_terminal_time(start_time, stop_event, func_name=f'{method}: capnobase_main - chunked')
	else:
		time_count.stop_terminal_time(start_time, stop_event, func_name=f'{method}: capnobase_main')
