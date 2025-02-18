from . import constants as C
import numpy as np

def estimation(ppg_signal, fs):
	pass

def evaluate(quality_arr, ref_quality, method='my', database='CB'):
	"""
	Calculate the quality of the provided PPG signal.

	Args:
		quality_arr: The quality array of the signal.
		ref_quality [int]: The reference quality
		method: 'my' or 'orphanidou'

	Returns:
		avg_quality: The average quality of the signal.
		diff_quality: The difference between the average quality and the reference quality (if provided).
	"""
	if method == 'my':
		avg_quality = np.mean(quality_arr)
	elif method == 'orphanidou':
		avg_quality = np.mean(quality_arr)
	else:
		raise ValueError(C.INVALID_QUALITY_METHOD)

	if database == 'BUT':
		rounded_q = 1 if avg_quality >= C.CORRELATION_THRESHOLD else 0
		diff_quality = abs(rounded_q - ref_quality)
		C.DIFF_QUALITY_SUM += diff_quality

		return avg_quality, diff_quality

	return avg_quality, None