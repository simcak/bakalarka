from bcpackage.butpackage import but_data, but_show
from bcpackage import preprocess, peaks, calcul, export, quality, globals as G, time
import neurokit2 as nk
import numpy as np

def but_ppg_main(method: str, show=False, first=False):
	"""
	Function to run the BUT PPG analysis.
	"""
	# Init before the loop
	G.DIFF_HR_LIST, G.DIFF_HR_LIST_QUALITY, G.DIFF_QUALITY_SUM = [], [], 0
	start_time = time.terminal_time()

	for j in range(3888):
		but_signal_info = but_data.extract_big(j, export=True)
		# filtered_ppg_signal = preprocess.filter_signal(but_signal_info['PPG_Signal'], but_signal_info['PPG_fs'])
		# but_signal_info['PPG_Peaks'] = peaks.detect_peaks(filtered_ppg_signal, but_signal_info['PPG_fs'])

		# import matplotlib.pyplot as plt
		# plt.plot(but_signal_info['PPG_Signal'])
		# plt.scatter(but_signal_info['PPG_Peaks'], but_signal_info['PPG_Signal'][but_signal_info['PPG_Peaks']], color='red')
		# plt.scatter(but_signal_info['QRS'], but_signal_info['PPG_Signal'][but_signal_info['QRS']], color='green')
		# plt.show()

	time.stop_terminal_time(start_time)
	exit()

	for i in range(G.BUT_DATA_LEN):
		id, fs, ref_quality, ref_hr, ppg_signal = but_data.extract(i)

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
			quality_info = quality.evaluate(filtered_ppg_signal, detected_peaks, fs,
								   None, ref_quality,
								   method='my_morpho', database='BUT')
			quality_info['Ref Q.'] = ref_quality
			name = 'My'
		
		# Execute NeuroKit library with:
		# 	Elgendi method for peak detection
		# 	Orphanidou method for quality estimation == templatematch
		elif method == 'neurokit':
			nk_signals, info = nk.ppg_process(preprocess.standardize_signal(ppg_signal), sampling_rate=fs, method="elgendi")
			detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
			quality_info = quality.evaluate(None, detected_peaks, fs,
								   nk_signals['PPG_Quality'], ref_quality,
								   method='orphanidou', database='BUT')
			quality_info['Ref Q.'] = ref_quality
			name = 'NK'
		
		else:
			raise ValueError(G.INVALID_METHOD)

		# Calculate the heart rate
		hr_info = calcul.heart_rate(detected_peaks, ref_hr, fs)
		if quality_info['Ref Q.'] == 1:
			G.DIFF_HR_LIST_QUALITY.append(hr_info['Diff HR'])

		export.to_csv_local(id, 8, i, hr_info, quality_info, None,
					  type=name, database='BUT', first=first)

		############################################### For testing purposes ##############################################
		if method == 'my' and show:
			but_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, detected_peaks, hr_info, id, i)
		elif method == 'neurokit' and show:
			but_show.neurokit_show(nk_signals, info, i)
		###################################################################################################################

	# Global results - outsinde the loop
	export.to_csv_global('all', None, None, type=name, database='BUT')
	time.stop_terminal_time(start_time)