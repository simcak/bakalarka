from bcpackage import globals as G

import matplotlib.pyplot as plt
import neurokit2 as nk

def neurokit_show(signals, info, i):
	"""
	fancy plot
	"""
	nk.ppg_plot(signals, info)
	plt.show()

def test_hub(ppg_signal, filtered_ppg_signal, our_peaks, hr_info, but_id, i):
	"""
	Here we choose which function for showing we want to use.
	"""
	if i >= G.SAMPLE_NUMBER_BUT:
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
	time = [i / 30 for i in range(len(signal1))]
	plt.figure(figsize=(14.4, 6))
	plt.plot(time, signal1, label='původní signál')
	plt.plot(time, signal2, label='filtrovaný signál')
	plt.scatter([time[p] for p in peaks], signal2[peaks], c='g', label='detekované vrcholy')
	plt.title(f'BUT PPG (id.: {but_id}) Referenční kvalita: 1', fontsize=17)
	plt.xlabel('Čas [s]', fontsize=14)
	plt.ylabel('Relativní amplituda', fontsize=14)
	plt.tight_layout()
	plt.legend(fontsize=14, loc='upper right')
	plt.show()