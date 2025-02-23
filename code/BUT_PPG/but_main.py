from bcpackage.butpackage import but_data, but_show
from bcpackage import preprocess, peaks, calcul, export, quality, globals as G
import neurokit2 as nk
import numpy as np

def but_ppg_main(method: str, show=False):
	"""
	Function to run the BUT PPG analysis.
	"""
	# Init before the loop
	G.DIFF_HR_LIST, G.DIFF_HR_LIST_QUALITY, G.DIFF_QUALITY_SUM = [], [], 0

	for i in range(G.BUT_DATA_LEN):
		id, fs, ref_quality, ref_hr, ppg_signal = but_data.extract(i, export=False)

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

		export.to_csv_local(id, i, hr_info, quality_info,
					  None, None, None, None, None,
					  type=name, database='BUT')

		############################################### For testing purposes ##############################################
		if method == 'my' and show:
			but_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, detected_peaks, hr_info, id, i)
		elif method == 'neurokit' and show:
			but_show.neurokit_show(nk_signals, info, i)

		if method == 'my':
			print('|  i\t|   ID\t\t|    Ref HR\t|    Our HR\t|    Diff HR\t|    Ref. Q\t|   Our Quality\t|')
		elif method == 'neurokit':
			print('|  i\t|   ID\t\t|    Ref HR\t|    NK HR\t|    Diff HR\t|    Ref. Q\t| Orph. Quality\t|')
		print(f'|  {i}\t|   {id}\t|    {round(hr_info["Ref HR"], 3)} bpm\t|   {round(hr_info["Calculated HR"], 3)} bpm\t|   {round(hr_info["Diff HR"], 3)} bpm\t|    {quality_info["Ref Q."]}\t\t|     {round(quality_info["Calc Q."], 3)}\t|')
		print('---------------------------------------------------------------------------------------------------------')
		###################################################################################################################

	# Global results - outsinde the loop
	export.to_csv_global(f'BUT {name} global',
					  None, None,
					  type=name, database='BUT')
	print('#########################################################################################################')