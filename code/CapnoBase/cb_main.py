from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, export
from bcpackage import constants as C

import neurokit2 as nk
import numpy as np

def capnobase_main(method: str, show = False):
	"""
	Function to run the CapnoBase analysis.
	0. Initialize the lists for the global results with the empty lists.
	1. Load the list of files.
	FOR LOOP:
		2. Iterate over the files.
		3. Extract the CapnoBase data.
		4. Preprocess the PPG signal = filtering = standardization + noise removal.
			- My method
			- Elgendi method
		5. Detect peaks in the preprocessed signal.
		6. Calculate the heart rate from the detected peaks (using IBI) + calculate the diff with the ref HR.
		7. Confusion matrix: TP, FP, FN.
		8. Performance metrics: Sensitivity, Precision.
		9. Export the data of each signal to a CSV file "./results.csv".

	10. Calculate the global results: Total Sensitivity and Precision + sum of FP, FN, TP + average HR diff.

	Args:
		method (str): The method to execute. Use either "my" or "elgendi".
		show (bool): Flag to show debugging or testing info.

	Returns:
		None (exports the results to a CSV file)
	"""
	C.TP_LIST, C.FP_LIST, C.FN_LIST, C.DIFF_HR_LIST = [], [], [], []
	capnobase_files = cb_data.list_of_files()

	for i in range(len(capnobase_files)):
		# Extract the data
		id, fs, ppg_signal, ref_peaks, ref_hr = cb_data.extract(capnobase_files[i])

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
			name = 'CB my'

		# Execute the Elgendi method
		elif method == 'elgendi':
			signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs, method="elgendi")	# Return: PPG_Raw  PPG_Clean  PPG_Rate  PPG_Quality  PPG_Peaks for each sample
			detected_peaks = np.where(signals['PPG_Peaks'] == 1)[0]
			name = 'CB elgendi'

		else:
			raise ValueError('Invalid method provided. Use either "my" or "elgendi".')

		# Calculate the heart rate
		our_hr, diff_hr = calcul.heart_rate(detected_peaks, ref_hr, fs)

		# Confusion matrix
		tp, fp, fn = calcul.confusion_matrix(detected_peaks, ref_peaks, tolerance=30)

		# Performance metrics
		local_sensitivity, local_precision = calcul.performance_metrics(tp, fp, fn)

		# Export the data
		export.to_csv_local(id, ref_hr, our_hr, diff_hr, i,
					  tp, fp, fn,
					  local_sensitivity, local_precision,
					  None, type=name)

		########################## For testing purposes ##########################
		if method == 'my' and show:
			cb_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, ref_peaks, detected_peaks, ref_hr, our_hr, capnobase_files[i], i)
		elif method == 'elgendi' and show:
			cb_show.neurokit_show(signals, info, i)
		print(f'{i}: File: {id} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm \t\t| Diff: {round(diff_hr, 3)} bpm')
		##########################################################################

	# Global results - outside the loop
	total_sensitivity = np.sum(C.TP_LIST) / (np.sum(C.TP_LIST) + np.sum(C.FN_LIST))
	total_precision = np.sum(C.TP_LIST) / (np.sum(C.TP_LIST) + np.sum(C.FP_LIST))
	# Export global
	export.to_csv_global((f'{name} global'),
					  np.average(C.DIFF_HR_LIST), None,
					  np.sum(C.TP_LIST), np.sum(C.FP_LIST), np.sum(C.FN_LIST),
					  total_sensitivity, total_precision)