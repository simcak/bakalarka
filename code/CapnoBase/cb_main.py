from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, SeP, export
import numpy as np

# from refpackage import aboy, elgendi

def capnobase_main():
	"""
	Function to run the CapnoBase analysis.
	1. Load the list of files.
	FOR LOOP:
		2. Iterate over the files.
		3. Extract the CapnoBase data.
		4. Preprocess the PPG signal = filtering = standardization + noise removal.
		5. Detect peaks in the preprocessed signal.
		6. Calculate the heart rate from the detected peaks (using IBI) + calculate the diff with the ref HR.
		7. Confusion matrix: TP, FP, FN.
		8. Performance metrics: Sensitivity, Precision.
		9. Export the data of each signal to a CSV file "./results.csv".

	10. Calculate the global results: Total Sensitivity and Precision + sum of FP, FN, TP + average HR diff.

	Args:
		None (uses the CapnoBase data)

	Returns:
		None (exports the results to a CSV file)
	"""
	capnobase_files = cb_data.list_of_files()
	tp_list = []
	fp_list = []
	fn_list = []
	diff_hr_list = []

	for i in range(len(capnobase_files)):
		id = capnobase_files[i][16:20]
		fs, ppg_signal, ref_peaks, ref_hr = cb_data.extract(capnobase_files[i], export=False)
		filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
		our_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
		# Calculate the heart rate
		our_hr = calcul.heart_rate(our_peaks, fs)
		diff_hr = abs(ref_hr - our_hr)
		diff_hr_list.append(diff_hr)

		# Confusion matrix
		tp, fp, fn = SeP.confusion_matrix(our_peaks, ref_peaks, tolerance=30)
		tp_list.append(tp)
		fp_list.append(fp)
		fn_list.append(fn)

		# Performance metrics
		local_sensitivity, local_precision = SeP.performance_metrics(tp, fp, fn)
		export.to_csv_local(id, ref_hr, our_hr, diff_hr, i,
					  tp, fp, fn,
					  local_sensitivity, local_precision,
					  quality=None, type='capnobase')

		############# For testing purposes #############
		# cb_show.test_hub(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, ref_hr, our_hr, capnobase_files[i], i)
		print(f'{i}: File: {id} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm \t\t| Diff: {round(diff_hr, 3)} bpm')
		################################################

	# Global results - outsinde the loop
	total_sensitivity = np.sum(tp_list)/(np.sum(tp_list)+np.sum(fn_list))
	total_precision = np.sum(tp_list)/(np.sum(tp_list)+np.sum(fp_list))
	export.to_csv_global('global capno',
					  np.average(diff_hr_list), None,
					  np.sum(tp_list), np.sum(fp_list), np.sum(fn_list),
					  total_sensitivity, total_precision)