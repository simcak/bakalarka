# Filter functions crossroad
def filter_signal(ppg_signal, capnobase_fs):
	# denoised_signal = remove_noise(ppg_signal, capnobase_fs, lowcut=0.5, highcut=5.0)
	# no_BLD_signal = remove_baseline_drift(denoised_signal, capnobase_fs, lowcut=0.5, highcut=5.0)
	# remove motion artifacts?
	# output_signal = no_BLD_signal
	output_signal = ppg_signal

	return output_signal