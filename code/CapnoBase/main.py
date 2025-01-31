from mypackage import data, preprocess, show

def main():
	capnobase_files = data.list_of_files()

	for i in range(len(capnobase_files)):
		capnobase_fs, ref_peaks, ppg_signal, ref_hr = data.extract(capnobase_files[i], export=False)
		print(f'File: {capnobase_files[i]} | Sampling rate: {capnobase_fs} Hz | Reference HR: {ref_hr} bpm')
		filtered_ppg_signal = preprocess.filter_signal(ppg_signal, capnobase_fs)
		# our_peaks = detect_peaks(filtered_ppg_signal, capnobase_fs)
		# interbeat_intervals = compute_interbeat_intervals(ref_peaks, capnobase_fs)
		# heart_rate = compute_heart_rate(interbeat_intervals)

	show.original_data(ppg_signal, ref_peaks, capnobase_files[0])

if __name__ == "__main__":
	main()