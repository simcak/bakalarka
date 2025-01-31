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
	ibi : array
		Interbeat intervals.
	"""
	ibi = np.diff(peaks) / fs
	return ibi

def heart_rate(peaks, ppg_signal, fs):
	"""
	Compute heart rate from the peaks.
	"""
	ibi = interbeat_intervals(peaks, fs)
	hr = 60 / np.mean(ibi)
	return hr