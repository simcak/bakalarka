import numpy as np

def police(but_signal_info: dict, i: int, print_err=False) -> bool:
	"""
	Function to check if the signal is valid.
	"""
	corrupted_signals_1 = [114, 155, 170, 176, 189, 190, 191, 192, 193, 204, 243, 244, 245, 246, 263, 280, 298,
						312, 318, 344, 346, 352, 353, 361, 368, 386, 395, 409, 423, 425, 463, 496,
						504, 514, 533, 568, 571, 587, 617, 617, 631, 655, 656, 695, 730, 731, 791,
						825, 851, 860, 861, 862, 863, 883, 886, 888, 904, 941, 976, 979,
						1030, 1062, 1066, 1070, 1084, 1103, 1120, 1138, 1142, 1157, 1192, 1237, 1251,
						1300, 1333, 1336, 1354, 1355, 1409, 1414, 1438, 1449, 1450, 1469, 1494, 1504, 1544, 1547,
						1615, 1664, 1669, 1723, 1739, 1760, 1761, 1763, 1773, 1774, 1777, 1779, 1780, 1781, 1782,
						1806, 1831, 1885, 1922, 1930, 1939, 1940, 1955, 1974, 1990, 1993, 2009, 2135, 2178, 2245,
						2349, 2455, 2465, 2515, 2527, 2557, 2613, 2729, 2891, 2945, 3106, 3161, 3210, 3213, 3322,
						3431, 3444, 3510, 3514, 3525, 3527, 3584, 3631, 3633, 3693, 3731, 3739, 3742, 3797, 3851]
	corrupted_signals_2 = [337, 908, 3635]
	corrupted_signals_3 = [81, 248, 278, 295, 306, 345, 362, 373, 488, 507, 510, 546, 590, 688, 779, 806, 854, 906,
						1064, 1118, 1126, 1137, 1172, 1179, 1226, 1294, 1502, 1513, 1601, 1637, 1644, 1746, 1808, 1912, 1972,
						2019, 2083, 2102, 2265, 2271, 2339, 2424, 2426, 2587, 3309, 3430, 3887]
	corrupted_signals_4 = [186, 277, 311, 313, 460, 471, 494, 498, 677, 846, 1088, 1215, 1289, 1349, 1546, 1696, 1755, 3701] # 313 is quality 1

	if but_signal_info['PPG_Signal'] is None:
		if print_err:
			print(f'\nNo PPG signal: \033[91m File {but_signal_info["ID"]} skipped.\033[0m')
		return True

	centered_signal = but_signal_info['PPG_Signal'] - np.mean(but_signal_info['PPG_Signal'])
	if np.std(centered_signal) == 0:
		if print_err:
			print("\nPREPROCESS =\033[91m Standard deviation of the signal is zero, cannot standardize.\033[0m")
		return True

	# Skip the files with in the 'i' iteration becase NeuroKit library cannot process it (plus it has low quality anyway)
	if (i in corrupted_signals_1):
		if print_err:
			print('\nFile "/Users/peta/.pyenv/versions/3.10.0/lib/python3.10/site-packages/neurokit2/epochs/epochs_create.py", line 164, in epochs_create')
			print('\t epoch_max_duration = int(max((i * sampling_rate for i in parameters["duration"])))')
			print(f'ValueError:\033[91m cannot convert float NaN to integer \nFile {but_signal_info["ID"]} skipped.\033[0m')
		return True
	
	if (i in corrupted_signals_2):
		if print_err:
			print('\nFile "/Users/peta/.pyenv/versions/3.10.0/lib/python3.10/site-packages/neurokit2/misc/listify.py", line 39, in _multiply_list')
			print('\t q, r = divmod(length, len(lst))')
			print(f'ZeroDivisionError:\033[91m integer division or modulo by zero \nFile {but_signal_info["ID"]} skipped.\033[0m')
		return True
	
	if (i in corrupted_signals_3):
		if print_err:
			print('\n/Users/peta/.pyenv/versions/3.10.0/lib/python3.10/site-packages/numpy/core/fromnumeric.py:3432: RuntimeWarning: Mean of empty slice.')
			print('return _methods._mean(a, axis=axis, dtype=dtype')
			print('/Users/peta/.pyenv/versions/3.10.0/lib/python3.10/site-packages/numpy/core/_methods.py:190: RuntimeWarning: invalid value encountered in double_scalars')
			print('ret = ret.dtype.type(ret / rcount)')
			print(f'RuntimeWarning:\033[91m invalid value encountered in double_scalars \nFile {but_signal_info["ID"]} skipped.\033[0m')
		return True
	
	if (i in corrupted_signals_4):
		if print_err:
			print('\nProblem with 2 order filtering in epochs_create.py')
			print(f'\033[91m File {but_signal_info["ID"]} skipped.\033[0m')
		return True

	return False