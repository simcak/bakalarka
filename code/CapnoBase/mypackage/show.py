import os
import matplotlib.pyplot as plt

# Plot the PPG signal with its peaks
def one_signal_peaks(ppg_signal, ref_peaks, capnobase_file):
	"""
	Plot the PPG signal with its peaks.
	"""
	plt.plot(ppg_signal)
	plt.scatter(ref_peaks, ppg_signal[ref_peaks], c='r')
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
	plt.plot(signal1)
	plt.scatter(peaks1, signal1[peaks1], c='r')
	plt.plot(signal2)
	plt.scatter(peaks2, signal2[peaks2], c='g')
	plt.title(f'Two Signals with Peaks (mat file num.: {os.path.basename(capnobase_file)[0:4]})')
	plt.xlabel('Samples')
	plt.ylabel('Signal')
	plt.show()