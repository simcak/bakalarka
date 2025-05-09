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

def test_hub(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, hr_info, capnobase_file, i):
	"""
	Here we choose which function for showing we want to use.
	"""
	if i >= G.SAMPLE_NUMBER_CB:
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

def two_signals_peaks(signal_raw, signal_filtr, ref_peaks, peaks_our, capnobase_file):
	"""
	Plot two signals with their peaks.
	"""
	fs = 300

	plt.figure(figsize=(14.4, 6))
	plt.title(f'CapnoBase ID: {os.path.basename(capnobase_file)[0:4]}', fontsize=16)
	plt.plot(signal_raw, label='Původní signál', color='#E23F44')
	plt.plot(signal_filtr, label='Filtrovaný signál', color='black', alpha=0.5)
	plt.scatter(peaks_our, signal_filtr[peaks_our], c='#02CCFF', label='Detekované vrcholy')
	plt.scatter(ref_peaks, signal_filtr[ref_peaks], c='black', label='Referenční vrcholy', marker='x', alpha=0.5)
	plt.xticks(ticks=range(0, len(signal_raw), 5 * fs), labels=[x // fs for x in range(0, len(signal_raw), 5 * fs)])
	plt.xticks(ticks=range(0, len(signal_raw) + 1, fs), minor=True)
	plt.tick_params(axis='both', which='major', labelsize=12)
	plt.xlabel('Čas [s]', fontsize=14)
	plt.ylabel('Amplituda', fontsize=14)
	plt.legend(loc='upper right', fontsize=14)
	plt.grid()
	plt.tight_layout()
	plt.show()