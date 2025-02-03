from mypackage import data, preprocess, show, peaks, calcul
# from refpackage import aboy, elgendi

def main_calculation(capnobase_file, show_plots=False):
	capnobase_fs, ppg_signal, ref_peaks, ref_ibi, ref_hr = data.extract(capnobase_file, export=False)
	filtered_ppg_signal = preprocess.filter_signal(ppg_signal, capnobase_fs)
	our_peaks = peaks.detect_peaks(filtered_ppg_signal, capnobase_fs)
	our_hr = calcul.heart_rate(our_peaks, capnobase_fs)

	print(f'File: {capnobase_file[6:]} | Ref HR: {round(ref_hr, 3)} bpm | Our HR: {round(our_hr, 3)} bpm\
	| Diff: {round(abs(ref_hr - our_hr), 3)} bpm')

	if show_plots:
		# show.one_signal_peaks(filtered_ppg_signal, our_peaks, capnobase_file)
		show.two_signals_peaks(ppg_signal, filtered_ppg_signal, ref_peaks, our_peaks, capnobase_file)
		# show.two_signals(ppg_signal, filtered_ppg_signal, capnobase_file)

	return ref_hr, our_hr

def main():
	capnobase_files = data.list_of_files()

	for i in range(len(capnobase_files)):
		print(f'{i})', end=' ')
		ref_hr, our_hr = main_calculation(capnobase_files[i], show_plots=False)

	# For testing purposes
	print('\n')
	file_index = 2
	ref_hr, our_hr = main_calculation(capnobase_files[file_index], show_plots=True)

if __name__ == "__main__":
	main()