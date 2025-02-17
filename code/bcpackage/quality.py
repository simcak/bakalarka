"""
Calculate the quality of the provided PPG signal.
Use our own method and the Orphanidou method for estimating the quality.
"""

import neurokit2 as nk
import numpy as np

def evaluate(ppg_signal, fs, method='my'):
	"""
	Calculate the quality of the provided PPG signal.

	Args:
		ppg_signal: 'my' = ___
		fs: sampling frequency
		method: 'my' or 'orphanidou'

	Returns:
		quality: quality of the PPG signal
	"""
	if method == 'my':
		quality = np.mean(ppg_signal)

	return quality