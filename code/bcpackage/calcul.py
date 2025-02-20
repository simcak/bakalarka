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

def confusion_matrix(our_peaks, ref_peaks, tolerance, add_to_list=True):
	"""
	Compute confusion matrix for the detected peaks.

	Args:
		our_peaks: list of detected peaks
		ref_peaks: list of reference peaks
		tolerance: num of samples that a peak can be off by && still be considered a match
		add_to_list: whether to add the results to the global lists

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

	if add_to_list:
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

def heart_rate(peaks, hr_in, fs, type='median'):
	"""
	Compute heart rate from the peaks.

	Args:
		peaks: list of peak indices
		hr_in: reference heart rate
		fs: sampling frequency
		type: 'mean' or (default) 'median'

	Returns:
		hr_out: our heart rate
		diff_hr: difference between our and reference heart rate
	"""
	our_ibi = interbeat_intervals(peaks, fs)

	if type == 'mean':
		hr_out = 60 / np.mean(our_ibi)
	elif type == 'median':
		hr_out = 60 / np.median(our_ibi)
	else:
		raise ValueError('Invalid type provided. Use either "mean" or "median".')

	if hr_in:
		diff_hr = abs(hr_in - hr_out)
		G.DIFF_HR_LIST.append(diff_hr)
	else:
		diff_hr = None


	return hr_out, diff_hr