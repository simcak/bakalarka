from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, SeP, export
import numpy as np

# from refpackage import aboy, elgendi

def capnobase_main():
	capnobase_files = cb_data.list_of_files()
	tp_list = []
	fp_list = []
	fn_list = []

	for i in range(len(capnobase_files)):
		print(f'{i})', end=' ')
		capnobase_fs, ppg_signal, ref_peaks, ref_hr = cb_data.extract(capnobase_files[i], export=False)
		filtered_ppg_signal = preprocess.filter_signal(ppg_signal, capnobase_fs)
		our_peaks = peaks.detect_peaks(filtered_ppg_signal, capnobase_fs)
		our_hr = calcul.heart_rate(our_peaks, capnobase_fs)
		diff_hr = round(abs(ref_hr - our_hr), 3)
		id = capnobase_files[i][16:20]

		print(f'File: {id} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm\
		| Diff: {diff_hr} bpm')

		# Confusion matrix
		tp, fp, fn = SeP.confusion_matrix(our_peaks, ref_peaks, tolerance=30)
		tp_list.append(tp)
		fp_list.append(fp)
		fn_list.append(fn)
		print(f'TP: {tp} | FP: {fp} | FN: {fn}')

		# Sensitivity, specificity, and accuracy
		local_sensitivity = tp/(tp+fn)
		local_positive_predictivity = tp/(tp+fp)
		print(f'Sensitivity: {round(local_sensitivity, 3)} | Positive Predictivity: {round(local_positive_predictivity, 3)}')

		# Export to CSV
		export.to_csv_local(id, tp, fp, fn, local_sensitivity, local_positive_predictivity, ref_hr, our_hr, round(abs(ref_hr - our_hr), 3), i)
		# For testing purposes
		# cb_show.test_hub(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, ref_hr, our_hr, capnobase_files[i], i)

	print('\n')
	print(f'TP: {np.sum(tp_list)} | FP: {np.sum(fp_list)} | FN: {np.sum(fn_list)}')
	total_sensitivity = np.sum(tp)/(np.sum(tp)+np.sum(fn))
	total_positive_predictivity = np.sum(tp)/(np.sum(tp)+np.sum(fp))
	print(f'Sensitivity: {total_sensitivity} | Positive Predictivity: {total_positive_predictivity}')
	export.to_csv_global('capnobase', np.sum(tp_list), np.sum(fp_list), np.sum(fn_list), total_sensitivity, total_positive_predictivity)