import numpy as np
from scipy.signal import find_peaks

def detect_peaks(ppg_signal, capnobase_fs):
	"""
	Detect peaks in the PPG signal.
	"""
	# Find peaks in the PPG signal
	peaks, _ = find_peaks(ppg_signal, distance=capnobase_fs/2)
	return peaks