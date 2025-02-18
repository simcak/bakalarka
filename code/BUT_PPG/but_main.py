from bcpackage.butpackage import but_data, but_show
from bcpackage import preprocess, peaks, calcul, export, constants as C, quality
import neurokit2 as nk
import numpy as np

def but_ppg_main(method: str, show=False):
	"""
	Function to run the BUT PPG analysis.
	"""
	# Init before the loop
	diff_hr_list, diff_hr_list_quality, C.DIFF_QUALITY_SUM = [], [], 0

	for i in range(C.BUT_DATA_LEN):
		id, fs, ref_quality, ref_hr, ppg_signal = but_data.extract(i, export=False)

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
			measured_quality, diff_quality = quality.evaluate([0.0, 0.0, 0.0], ref_quality,
													 method=method, database='BUT')
			name = 'My'
		
		# Execute NeuroKit library with:
		# 	Elgendi method for peak detection
		# 	Orphanidou method for quality estimation == templatematch
		elif method == 'neurokit':
			nk_signals, info = nk.ppg_process(preprocess.standardize_signal(ppg_signal), sampling_rate=fs, method="elgendi")
			detected_peaks = np.where(nk_signals['PPG_Peaks'] == 1)[0]
			measured_quality, diff_quality = quality.evaluate(nk_signals['PPG_Quality'], ref_quality,
													 method='orphanidou', database='BUT')
			name = 'NK'
		
		else:
			raise ValueError(C.INVALID_METHOD)

		# Calculate the heart rate
		calculated_hr, diff_hr = calcul.heart_rate(detected_peaks, ref_hr, fs)
		diff_hr_list.append(diff_hr)
		if ref_quality == 1:
			diff_hr_list_quality.append(diff_hr)

		export.to_csv_local(id, i, ref_hr, calculated_hr, diff_hr,
					  None, None, None, None, None,
					  ref_quality, measured_quality, diff_quality,
					  type=name, database='BUT')

		############################################### For testing purposes ##############################################
		if method == 'my' and show:
			but_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, detected_peaks, ref_hr, calculated_hr, id, i)
		elif method == 'neurokit' and show:
			but_show.neurokit_show(nk_signals, info, i)

		if method == 'my':
			print('|  i\t|   ID\t\t|    Ref HR\t|    Our HR\t|    Diff HR\t|    Ref. Q\t|   Our Quality\t|')
		elif method == 'neurokit':
			print('|  i\t|   ID\t\t|    Ref HR\t|    NK HR\t|    Diff HR\t|    Ref. Q\t| Orph. Quality\t|')
		print(f'|  {i}\t|   {id}\t|    {round(ref_hr, 3)} bpm\t|   {round(calculated_hr, 3)} bpm\t|   {round(diff_hr, 3)} bpm\t|    {ref_quality}\t\t|     {round(measured_quality, 3)}\t|')
		print('---------------------------------------------------------------------------------------------------------')
		###################################################################################################################

	# Global results - outsinde the loop
	export.to_csv_global(f'BUT {name} global',
					  np.average(diff_hr_list), np.average(diff_hr_list_quality), C.DIFF_QUALITY_SUM, None,
					  None, None, None, None, None,
					  type=name, database='BUT')
	print('---------------------------------------------------------------------------------------------------------')