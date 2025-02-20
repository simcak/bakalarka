from bcpackage import globals as G

import os
import matplotlib.pyplot as plt
import neurokit2 as nk

def neurokit_show(signals, info, i):
	"""
	fancy plot
	"""
	if i == G.SAMPLE_NUMBER_CB:
		nk.ppg_plot(signals, info)
		plt.show()

def test_hub(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, ref_hr, our_hr, capnobase_file, i):
	"""
	Here we choose which function for showing we want to use.
	"""
	if i == G.SAMPLE_NUMBER_CB:
		# one_signal_peaks(filtered_ppg_signal, our_peaks, capnobase_file)
		two_signals_peaks(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, capnobase_file)
		# two_signals(ppg_signal, filtered_ppg_signal, capnobase_file)

# Plot the PPG signal with its peaks
def one_signal_peaks(ppg_signal, peaks, capnobase_file):
	"""
	Plot the PPG signal with its peaks.
	"""
	plt.plot(ppg_signal)
	plt.scatter(peaks, ppg_signal[peaks], c='r')
	plt.title(f'PPG Signal with Peaks (mat file num.: {os.path.basename(capnobase_file)[0:4]})')
	plt.xlabel('Samples')
	plt.ylabel('PPG Signal')
	plt.show()

def two_signals(signal1, signal2, capnobase_file):
	"""
	Plot two signals.
	"""
	plt.plot(signal1)
	plt.plot(signal2)
	plt.title(f'Two Signals (mat file num.: {os.path.basename(capnobase_file)[0:4]})')
	plt.xlabel('Samples')
	plt.ylabel('Signal')
	plt.show()

def two_signals_peaks(signal1, signal2, peaks1, peaks2, capnobase_file):
	"""
	Plot two signals with their peaks.
	"""
	plt.figure(figsize=(14.4, 6))
	plt.plot(signal1)
	plt.scatter(peaks1, signal1[peaks1], c='r')
	plt.plot(signal2)
	plt.scatter(peaks2, signal2[peaks2], c='g')
	plt.title(f'Two Signals with Peaks (mat file num.: {os.path.basename(capnobase_file)[0:4]})')
	plt.xlabel('Samples')
	plt.ylabel('Signal')
	plt.show()