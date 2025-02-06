from bcpackage.capnopackage import cb_data, cb_show
from bcpackage import preprocess, peaks, calcul
import numpy as np

# from refpackage import aboy, elgendi

def capnobase_main():
	capnobase_files = cb_data.list_of_files()
	sensitivity = []
	specificity = []
	accuracy = []

	for i in range(len(capnobase_files)):
		print(f'{i})', end=' ')
		capnobase_fs, ppg_signal, ref_peaks, ref_ibi, ref_hr = cb_data.extract(capnobase_files[i], export=False)
		filtered_ppg_signal = preprocess.filter_signal(ppg_signal, capnobase_fs)
		our_peaks = peaks.detect_peaks(filtered_ppg_signal, capnobase_fs)
		our_hr = calcul.heart_rate(our_peaks, capnobase_fs)

		print(f'File: {capnobase_files[i][6:]} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm\
		| Diff: {round(abs(ref_hr - our_hr), 3)} bpm')

		# For testing purposes
		cb_show.test_hub(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, ref_hr, our_hr, capnobase_files[i], i)

		# Sensitivity, specificity, and accuracy
		sensitivity.append(len(set(ref_peaks) & set(our_peaks)) / len(ref_peaks))
		specificity.append(len(set(ref_peaks) & set(our_peaks)) / len(our_peaks))
		accuracy.append(1 - np.mean(np.abs(ref_hr - our_hr) / ref_hr))

	print('\n')
	print(f'Sensitivity: {np.mean(sensitivity)} | Specificity: {np.mean(specificity)} | Accuracy: {np.mean(accuracy)}')