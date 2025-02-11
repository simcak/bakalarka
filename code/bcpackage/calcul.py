import numpy as np

def performance_metrics(tp, fp, fn):
	"""
	tp: true positive
	fp: false positive
	fn: false negative

	Returns: sensitivity, precision
	"""
	sensitivity = tp / (tp + fn)
	precision = tp / (tp + fp)
	return sensitivity, precision

def confusion_matrix(our_peaks, ref_peaks, tolerance):
	"""
	our_peaks: list of detected peaks
	ref_peaks: list of reference peaks
	tolerance: num of samples that a peak can be off by && still be considered a match

	Returns: TP, FP, FN

	Why don't we use logical and instead of bitwise and?
	- "and" "is used for logical conjunction between scalar boolean values
	- it is NOT used for element-wise operations on arrays
	"""
	tp = 0
	fp = 0
	fn = 0

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

	# If there are no reference peaks
	if len(ref_peaks) == 0:
		tp = 0
		fp = 0
		fn = 0

	return tp, fp, fn

def interbeat_intervals(peaks, fs):
	"""
	Compute interbeat intervals from reference peaks.
	
	Parameters
	----------
	ref_peaks : array
		Reference peaks.
	fs : int
		Sampling frequency.
	
	Returns
	-------
	ibi : array
		Interbeat intervals in seconds.
	"""
	ibi = np.diff(peaks) / fs
	return ibi

def heart_rate(peaks, fs):
	"""
	Compute heart rate from the peaks.
	"""
	ibi = interbeat_intervals(peaks, fs)
	hr = 60 / np.mean(ibi)
	return hr