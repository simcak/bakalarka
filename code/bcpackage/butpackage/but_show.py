import os
import matplotlib.pyplot as plt

def test_hub(ppg_signal, filtered_ppg_signal, our_peaks, ref_hr, our_hr, but_id, i):
	"""
	Here we choose which function for showing we want to use.
	"""
	if i == 39:
		# one_signal_peaks(filtered_ppg_signal, our_peaks, but_id)
		two_signals_peaks(ppg_signal, filtered_ppg_signal, our_peaks, but_id)
		# two_signals(ppg_signal, filtered_ppg_signal, but_id)

# Plot the PPG signal with its peaks
def one_signal_peaks(ppg_signal, peaks, but_id):
	"""
	Plot the PPG signal with its peaks.
	"""
	plt.plot(ppg_signal)
	plt.scatter(peaks, ppg_signal[peaks], c='r')
	plt.title(f'PPG Signal with Peaks (id.: {but_id})')
	plt.xlabel('Samples')
	plt.ylabel('PPG Signal')
	plt.show()

def two_signals(signal1, signal2, but_id):
	"""
	Plot two signals.
	"""
	plt.plot(signal1)
	plt.plot(signal2)
	plt.title(f'Two Signals (id.: {but_id})')
	plt.xlabel('Samples')
	plt.ylabel('Signal')
	plt.show()

def two_signals_peaks(signal1, signal2, peaks, but_id):
	"""
	Plot two signals with their peaks.
	"""
	plt.figure(figsize=(14.4, 6))
	plt.plot(signal1)
	plt.plot(signal2)
	plt.scatter(peaks, signal2[peaks], c='g')
	plt.title(f'Two Signals with Peaks (id.: {but_id})')
	plt.xlabel('Samples')
	plt.ylabel('Signal')
	plt.show()