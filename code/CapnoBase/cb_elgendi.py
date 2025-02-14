from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul, export
import matplotlib.pyplot as plt
import numpy as np

import neurokit2 as nk

def capnobase_elgendi():
	"""
	"""
	capnobase_files = cb_data.list_of_files()
	tp_list, fp_list, fn_list = [], [], []
	diff_hr_list = []

	for i in range(len(capnobase_files)):
		id, fs, ppg_signal, ref_peaks, ref_hr = cb_data.extract(capnobase_files[i])

		# Execute the Elgendi method
		signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs, method="elgendi")	# Return: PPG_Raw  PPG_Clean  PPG_Rate  PPG_Quality  PPG_Peaks
		elgendi_peaks = np.where(signals['PPG_Peaks'] == 1)[0]

		# Calculate the heart rate
		our_hr, diff_hr = calcul.heart_rate(elgendi_peaks, ref_hr, fs)
		diff_hr_list.append(diff_hr)

		# Confusion matrix
		tp, fp, fn = calcul.confusion_matrix(elgendi_peaks, ref_peaks, tolerance=30)
		tp_list.append(tp)
		fp_list.append(fp)
		fn_list.append(fn)

		# Performance metrics
		local_sensitivity, local_precision = calcul.performance_metrics(tp, fp, fn)

		export.to_csv_local(id, ref_hr, our_hr, diff_hr, i,
						tp, fp, fn,
						local_sensitivity, local_precision,
						None, type='capnobase_elgendi')

		############# For testing purposes #############
		# nk.ppg_plot(signals, info)
		# plt.show()
		print(f'{i}: File: {id} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm \t\t| Diff: {round(diff_hr, 3)} bpm')
		################################################

	# Global results - outsinde the loop
	total_sensitivity = np.sum(tp_list)/(np.sum(tp_list)+np.sum(fn_list))
	total_precision = np.sum(tp_list)/(np.sum(tp_list)+np.sum(fp_list))

	export.to_csv_global('CB elgendi',
						np.average(diff_hr_list), None,
						np.sum(tp_list), np.sum(fp_list), np.sum(fn_list),
						total_sensitivity, total_precision)

