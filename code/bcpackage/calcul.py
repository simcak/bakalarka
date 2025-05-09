from . import globals as G
import numpy as np

def performance_metrics(tp, fp, fn):
	"""
	Args:
		tp: true positive
		fp: false positive
		fn: false negative

	Returns:
		sensitivity: TP / (TP + FN)
		precision: TP / (TP + FP)
	"""
	sensitivity = tp / (tp + fn)
	precision = tp / (tp + fp)

	return sensitivity, precision

def confusion_matrix(our_peaks, ref_peaks, tolerance):
	"""
	Compute confusion matrix for the detected peaks.

	Args:
		our_peaks: list of detected peaks
		ref_peaks: list of reference peaks
		tolerance: num of samples that a peak can be off by && still be considered a match

	Returns:
		TP, FP, FN

	Why don't we use logical and instead of bitwise and?
	- "and" "is used for logical conjunction between scalar boolean values
	- it is NOT used for element-wise operations on arrays
	"""
	tp, fp, fn = 0, 0, 0

	# If there are no reference peaks
	if len(ref_peaks) == 0:
		return 0, 0, 0

	# Find false positives (FP)
	for i in range(len(our_peaks)):
		match = np.any((our_peaks[i] >= ref_peaks - tolerance) & (our_peaks[i] <= ref_peaks + tolerance))
		if not match:
			fp += 1

	# Find true positive (TP) and false negatives (FN)
	for i in range(len(ref_peaks)):
		matches = np.where((ref_peaks[i] >= our_peaks - tolerance) & (ref_peaks[i] <= our_peaks + tolerance))[0]
		if len(matches) == 0:
			fn += 1
		else:
			tp += 1
			if len(matches) >= 2:
				fp += len(matches) - 1

	G.TP_LIST.append(tp)
	G.FP_LIST.append(fp)
	G.FN_LIST.append(fn)

	return tp, fp, fn

def interbeat_intervals(peaks, fs):
	"""
	Compute interbeat intervals from reference peaks.

	Args:
		peaks: list of peak indices
		fs: sampling frequency

	Returns:
		ibi: list of interbeat intervals
	"""
	ibi = np.diff(peaks) / fs
	return ibi

def heart_rate(peaks, ref_hr, fs, init=False, math_method='median'):
	"""
	Compute heart rate from the peaks.

	Args:
		peaks: list of peak indices
		ref_hr: reference heart rate
		fs: sampling frequency
		math_method: 'mean' or (default) 'median'

	Returns:
		hr_info: dictionary containing calculated HR, reference HR, and difference HR
	"""
	def compute_sdnn(ibi):
		"""
		Compute SDNN from interbeat intervals.

		Args:
			ibi: list of interbeat intervals

		Returns:
			sdnn: standard deviation of the interbeat intervals
		"""
		sdnn = np.std(ibi)
		return sdnn
	def compute_rmssd(ibi):
		"""
		Compute RMSSD from interbeat intervals.

		Args:
			ibi: list of interbeat intervals

		Returns:
			rmssd: root mean square of successive differences
		"""
		rmssd = np.sqrt(np.mean(np.square(np.diff(ibi))))
		return rmssd
	def plot_histogram(ibi, fs):
		"""
		Plot histogram of interbeat intervals.

		Args:
			ibi: list of interbeat intervals
			fs: sampling frequency
		"""
		import matplotlib.pyplot as plt
		from scipy.stats import mode

		instant_hr = 60 / ibi
		median_hr = np.median(instant_hr)
		modus_hr = mode(instant_hr, nan_policy='omit').mode[0]
		mean_hr = np.mean(instant_hr)

		plt.hist(instant_hr, bins=20, color='black', alpha=0.5)
		plt.title('Histogram okamžitých srdečních frekvencí')
		plt.xlabel('Tepová frekvence (tepy za minutu)')
		plt.ylabel('Počet vzorků')
		plt.axvline(median_hr, color='red', linestyle='dashed', linewidth=2, label=f'Medián: {median_hr:.2f} bpm')
		plt.axvline(mean_hr, color='blue', linestyle='dashed', linewidth=2, label=f'Průměr: {mean_hr:.2f} bpm')
		plt.axvline(modus_hr, color='green', linestyle='dashed', linewidth=2, label=f'Modus: {modus_hr:.2f} bpm')
		plt.legend()
		plt.grid()
		plt.show()

	our_ibi = interbeat_intervals(peaks, fs)

	if math_method == 'mean':
		hr_out = 60 / np.mean(our_ibi)
	elif math_method == 'median':
		hr_out = 60 / np.median(our_ibi)	# median is more robust
	else:
		raise ValueError('Invalid math_method provided. Use either "mean" or "median".')

	# Comparing our detected peaks with ref_hr (BUT) or ref_peaks hr (CB)
	if init == False:
		diff_hr = abs(ref_hr - hr_out)
		sdnn = compute_sdnn(our_ibi)
		rmssd = compute_rmssd(our_ibi)
		G.DIFF_HR_LIST.append(diff_hr)
		# plot_histogram(our_ibi, fs)
	# For detecting HR from referential peaks (CB)
	elif init == True:
		ref_hr = hr_out
		hr_out, diff_hr, sdnn, rmssd = None, None, None, None

	hr_info = {
		'Calculated HR': hr_out,
		'Ref HR': ref_hr,
		'Diff HR': diff_hr,
		'SDNN': sdnn,
		'RMSSD': rmssd
	}

	return hr_info