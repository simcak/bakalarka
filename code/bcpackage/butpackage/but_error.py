import numpy as np

def police(but_signal_info: dict) -> bool:
	"""
	Function to check if the signal is valid.
	"""
	if but_signal_info['PPG_Signal'] is None:
		print(f'File {but_signal_info["ID"]} skipped.')
		return True

	centered_signal = but_signal_info['PPG_Signal'] - np.mean(but_signal_info['PPG_Signal'])
	if np.std(centered_signal) == 0:
		print("\033[91mPREPROCESS = Standard deviation of the signal is zero, cannot standardize.\033[0m")
		return True

	return False