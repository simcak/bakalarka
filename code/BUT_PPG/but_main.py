from bcpackage.butpackage import but_data, but_error, but_show
from bcpackage import preprocess, peaks, calcul, export, quality, globals as G, time_count, hjorth
import neurokit2 as nk
import numpy as np

def but_ppg_main(method: str, show=False, first=False):
	"""
	Function to run the BUT PPG analysis.
	"""
	# Init before the loop
	G.DIFF_HR_LIST, G.DIFF_HR_LIST_QUALITY, G.DIFF_QUALITY_SUM = [], [], 0
	start_time, stop_event = time_count.terminal_time()

	for i in range(G.BUT_DATA_LEN):
		but_signal_info = but_data.extract(i)
		if but_error.police(but_signal_info, i):
			if show:
				print(f"\033[91mSkipping index {i} due to invalid signal info.\033[0m")
			continue

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(but_signal_info['PPG_Signal'], but_signal_info['PPG_fs'])
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, but_signal_info['PPG_fs'])
			quality_info = quality.evaluate(filtered_ppg_signal, detected_peaks, but_signal_info['PPG_fs'],
								   None, but_signal_info['Ref_Quality'],
								   method='my_morpho', database='BUT')
			quality_info['Ref Q.'] = but_signal_info['Ref_Quality']
			# hjorth.compute_hjorth_parameters(but_signal_info['PPG_Signal'], but_signal_info['PPG_fs'], detected_peaks, but_signal_info['Ref_HR'], but_signal_info['ID'], quality=but_signal_info['Ref_Quality'])
			name = 'My'
		
		# Execute NeuroKit library with:
		# 	Elgendi method for peak detection
		# 	Orphanidou method for quality estimation == templatematch
		elif method == 'neurokit':
			nk_signals, info = nk.ppg_process(preprocess.standardize_signal(but_signal_info['PPG_Signal']), sampling_rate=but_signal_info['PPG_fs'], method="elgendi")
			detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
			quality_info = quality.evaluate(None, detected_peaks, but_signal_info['PPG_fs'],
								   nk_signals['PPG_Quality'], but_signal_info['Ref_Quality'],
								   method='orphanidou', database='BUT')
			quality_info['Ref Q.'] = but_signal_info['Ref_Quality']
			name = 'NK'

		# Calculate the heart rate
		hr_info = calcul.heart_rate(detected_peaks, but_signal_info['Ref_HR'], but_signal_info['PPG_fs'])
		if quality_info['Ref Q.'] == 1:
			G.DIFF_HR_LIST_QUALITY.append(hr_info['Diff HR'])

		export.to_csv_local(but_signal_info['ID'], 8, i, hr_info, quality_info, None,
					  type=name, database='BUT', first=first)

		############################################### For testing purposes ##############################################
		if method == 'my' and show:
			but_show.test_hub(preprocess.standardize_signal(but_signal_info['PPG_Signal']), filtered_ppg_signal, detected_peaks, hr_info, but_signal_info['ID'], i)
		elif method == 'neurokit' and show:
			but_show.neurokit_show(nk_signals, info, i)
		# print(f'File {i}: \nsignal:{but_signal_info["PPG_Signal"]}\nRef HR: {but_signal_info["Ref_HR"]}\nDetected peaks: {detected_peaks}\nHR info: {hr_info}')
		# print(f'File {i} done. ID: {but_signal_info["ID"]}, Ref Quality: {quality_info["Ref Q."]}')
		# print(f'HR info: {hr_info}')
		###################################################################################################################

	# Global results - outsinde the loop
	export.to_csv_global(None, None, type=name, database='BUT')
	time_count.stop_terminal_time(start_time, stop_event)