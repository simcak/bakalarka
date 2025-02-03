import os
import matplotlib.pyplot as plt

# Plot the PPG signal with its peaks
def one_signal(ppg_signal, ref_peaks, capnobase_file):
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