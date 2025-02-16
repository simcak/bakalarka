from bcpackage.butpackage import but_data, but_show
from bcpackage import preprocess, peaks, calcul, export, constants as C

import neurokit2 as nk
import scipy.io
import numpy as np

def size_of_but_database():
	mat_data = scipy.io.loadmat('./BUT_PPG/databases/BUT_PPG.mat')
	ppg_data = len(mat_data['BUT_PPG']['PPG'][0, 0])

	return ppg_data

################################################################################
def but_ppg_main(method: str, show = False):
	"""
	Function to run the BUT PPG analysis.
	"""
	diff_hr_list, diff_hr_list_quality = [], []

	for i in range(size_of_but_database()):
		id, fs, quality, ref_hr, ppg_signal = but_data.extract(i, export=False)

		# Execute my method
		if method == 'my':
			filtered_ppg_signal = preprocess.filter_signal(ppg_signal, fs)
			detected_peaks = peaks.detect_peaks(filtered_ppg_signal, fs)
			name = 'BUT my'
		
		# Execute the Elgendi method
		elif method == 'elgendi':
			signals, info = nk.ppg_process(ppg_signal, sampling_rate=fs, method="elgendi")
			detected_peaks = np.where(signals['PPG_Peaks'] == 1)[0]
			name = 'BUT elgendi'
		
		else:
			raise ValueError(C.INVALID_METHOD)

		# Calculate the heart rate
		our_hr, diff_hr = calcul.heart_rate(detected_peaks, ref_hr, fs)
		diff_hr_list.append(diff_hr)
		if quality:
			diff_hr_list_quality.append(diff_hr)

		export.to_csv_local(id, ref_hr, our_hr, diff_hr, i,
					  None, None, None, None, None,
					  quality=quality, type='but ppg')

		########################## For testing purposes ##########################
		if method == 'my' and show:
			but_show.test_hub(preprocess.standardize_signal(ppg_signal), filtered_ppg_signal, detected_peaks, ref_hr, our_hr, id, i)
		elif method == 'elgendi' and show:
			but_show.neurokit_show(signals, info, i)
		print(f'{i}: ID: {id} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm\t\t| Diff: {round(diff_hr, 3)} bpm\t| Quality: {quality}')
		##########################################################################

	# Global results - outsinde the loop
	export.to_csv_global(f'{name} global',
					  np.average(diff_hr_list), np.average(diff_hr_list_quality),
					  None, None, None, None, None)