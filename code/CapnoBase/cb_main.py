from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, export, quality, globals as G

import neurokit2 as nk
import numpy as np

def capnobase_main(method: str, show=False):
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
		id, fs, ppg_signal, ref_peaks, ref_hr = cb_data.extract(G.CB_FILES[i])

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
			measured_quality = 0.0
			G.QUALITY_LIST.append(measured_quality)
			name = 'My'

		# Execute the NeuroKit package with:
		#	Elgendi method for peak detection and
		#	Orphanidou method for quality estimation == templatematch
		elif method == 'neurokit':
			nk_signals, nk_info = nk.ppg_process(ppg_signal, sampling_rate=fs, method="elgendi", method_quality="templatematch")	# Return: PPG_Raw  PPG_Clean  PPG_Rate  PPG_Quality  PPG_Peaks for each sample
			detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
			measured_quality = np.mean(nk_signals['PPG_Quality'])
			G.QUALITY_LIST.append(measured_quality)
			name = 'NK'

		else:
			raise ValueError(G.INVALID_METHOD)

		calculated_hr, diff_hr = calcul.heart_rate(detected_peaks, ref_hr, fs)
		tp, fp, fn = calcul.confusion_matrix(detected_peaks, ref_peaks, tolerance=30)
		local_sensitivity, local_precision = calcul.performance_metrics(tp, fp, fn)

		export.to_csv_local(id, i,
					  ref_hr, calculated_hr, diff_hr,
					  tp, fp, fn,
					  local_sensitivity, local_precision,
					  None, measured_quality, None,
					  type=name, database='CB')

		################################### For testing purposes ##################################
		if method == 'my' and show:
			cb_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, ref_peaks, detected_peaks, ref_hr, calculated_hr, G.CB_FILES[i], i)
		elif method == 'neurokit' and show:
			cb_show.neurokit_show(nk_signals, nk_info, i)

		if method == 'my':
			print('|  i\t|  ID\t|    Sen\t|    Precision\t|    Diff HR\t|  Our Quality\t|')
		elif method == 'neurokit':
			print('|  i\t|  ID\t|    Sen\t|    Precision\t|    Diff HR\t| Orph. Quality\t|')
		print(f'|  {i}\t|  {id}\t|    {round(local_sensitivity, 3)}\t|    {round(local_precision, 3)}\t|   {round(diff_hr, 3)} bpm\t|     {round(measured_quality, 3)}\t|')
		print('---------------------------------------------------------------------------------')
		############################################################################################

	# Global results - outside the loop
	total_sensitivity = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FN_LIST))
	total_precision = np.sum(G.TP_LIST) / (np.sum(G.TP_LIST) + np.sum(G.FP_LIST))

	export.to_csv_global((f'CB {name} global'),
					  np.average(G.DIFF_HR_LIST), None, np.average(G.QUALITY_LIST),
					  np.sum(G.TP_LIST), np.sum(G.FP_LIST), np.sum(G.FN_LIST),
					  total_sensitivity, total_precision,
					  type=name, database='CB')
	print('#################################################################################')