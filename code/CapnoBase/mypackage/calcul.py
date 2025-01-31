import numpy as np

# ToFix
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
	interbeat_intervals : array
		Interbeat intervals.
	"""
	interbeat_intervals = np.diff(peaks) / fs
	return interbeat_intervals

def heart_rate(peaks, ppg_signal, fs):
	"""
	Compute heart rate from the peaks.
	"""
	hr = 60 * len(peaks) / (len(ppg_signal) / fs)
	return hr