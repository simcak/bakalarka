import h5py
import csv
from bcpackage import calcul

def list_of_files():
	"""
	Generate a list of CapnoBase .mat files.

	Returns:
		list: A list of file paths to CapnoBase .mat files.
	"""
	path = './CapnoBase/mat/'	# this path is from the main.py perspective
	capnobase_files = [
		f'{path}0329_8min.mat', f'{path}0328_8min.mat', f'{path}0030_8min.mat', f'{path}0031_8min.mat', f'{path}0322_8min.mat', f'{path}0133_8min.mat',
		f'{path}0018_8min.mat', f'{path}0125_8min.mat', f'{path}0370_8min.mat', f'{path}0128_8min.mat', f'{path}0015_8min.mat', f'{path}0333_8min.mat',
		f'{path}0332_8min.mat', f'{path}0122_8min.mat', f'{path}0123_8min.mat', f'{path}0009_8min.mat', f'{path}0148_8min.mat', f'{path}0149_8min.mat',
		f'{path}0311_8min.mat', f'{path}0142_8min.mat', f'{path}0325_8min.mat', f'{path}0134_8min.mat', f'{path}0150_8min.mat', f'{path}0127_8min.mat',
		f'{path}0309_8min.mat', f'{path}0147_8min.mat', f'{path}0038_8min.mat', f'{path}0105_8min.mat', f'{path}0104_8min.mat', f'{path}0032_8min.mat',
		f'{path}0103_8min.mat', f'{path}0035_8min.mat', f'{path}0313_8min.mat', f'{path}0312_8min.mat', f'{path}0016_8min.mat', f'{path}0330_8min.mat',
		f'{path}0331_8min.mat', f'{path}0121_8min.mat', f'{path}0029_8min.mat', f'{path}0028_8min.mat', f'{path}0115_8min.mat', f'{path}0023_8min.mat'
		]

	return capnobase_files

def extract(capnobase_file, export=False):
	"""
	Extract data from a CapnoBase .mat file.

	Args:
		capnobase_file (str): Path to the CapnoBase .mat file.
		export (bool): Whether to export the data to a CSV file or not. Non-mandatory.

	Returns:
		capnobase_fs (float): Sampling rate of the pleth signal.
		ref_peaks (numpy.ndarray): Array of reference peak positions.
		ppg_signal (numpy.ndarray): Array of pleth signal values.
		ref_hr (float): Reference heart rate calculated from the peaks and signal length.
	"""
	def export_file(output_file, capnobase_file, capnobase_fs, ref_peaks, ppg_signal):
		"""
		Export data to a CSV file.
		We use it for checking the data.
		"""
		with open(output_file, 'w', newline='') as csvfile:
			writer = csv.writer(csvfile)

			# File name
			writer.writerow(['File Name'])
			writer.writerow([capnobase_file[2:]])

			# Write capnobase_fs
			writer.writerow([])
			writer.writerow(['PPG_fs'])
			writer.writerow([capnobase_fs])

			# Write reference peaks
			writer.writerow([])
			writer.writerow(['PPG Peaks'])
			writer.writerow(ref_peaks)

			# Write ppg signals
			writer.writerow([])
			writer.writerow(['PPG Signal'])
			writer.writerow(ppg_signal)

	# Load the .mat file
	with h5py.File(capnobase_file, 'r') as mat_data:
		# Access the required datasets
		param = mat_data['param']
		labels = mat_data['labels']
		signal = mat_data['signal']

		# Extract specific data
		capnobase_fs = param['samplingrate']['pleth'][0][0].astype(int)
		ref_peaks = labels['pleth']['peak']['x'][:].astype(int).flatten()
		ppg_signal = signal['pleth']['y'][:].flatten()
		ref_hr, _ = calcul.heart_rate(ref_peaks, None, capnobase_fs)

		# Export data to a CSV file
		if export:
			output_file = f'./csv/capnobase_{capnobase_file[6:10]}.csv'
			export_file(output_file, capnobase_file, capnobase_fs, ref_peaks, ppg_signal)

	return capnobase_fs, ppg_signal, ref_peaks, ref_hr

