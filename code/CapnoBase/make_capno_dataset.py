import h5py
import csv

def extract_data(capnobase_file, output_file):
	# Load the .mat file
	with h5py.File(capnobase_file, 'r') as mat_data:
		# Access and extract the required datasets
		param = mat_data['param']
		labels = mat_data['labels']
		signal = mat_data['signal']

		# Extract specific data
		PPG_fs = param['samplingrate']['pleth'][0][0]
		PPG_peaks = labels['pleth']['peak']['x'][:].flatten()
		PPG_signal = signal['pleth']['y'][:].flatten()

	# Export the metadata and signals to a structured CSV file
	with open(output_file, 'w', newline='') as csvfile:
		writer = csv.writer(csvfile)

		# File name
		writer.writerow(['File Name'])
		writer.writerow([capnobase_file[2:]])

		# Write PPG_fs
		writer.writerow([])
		writer.writerow(['PPG_fs'])
		writer.writerow([PPG_fs])

		# Write PPG peaks
		writer.writerow([])
		writer.writerow(['PPG Peaks'])
		writer.writerow(PPG_peaks)

		# Write PPG signals
		writer.writerow([])
		writer.writerow(['PPG Signal'])
		writer.writerow(PPG_signal)

# Main function
def main():
	# Use: find . -iname "*.mat" -print0 | xargs -0 echo
	capnobase_files = [
		'./mat/0329_8min.mat', './mat/0328_8min.mat', './mat/0030_8min.mat', './mat/0031_8min.mat', './mat/0322_8min.mat', './mat/0133_8min.mat',
		'./mat/0018_8min.mat', './mat/0125_8min.mat', './mat/0370_8min.mat', './mat/0128_8min.mat', './mat/0015_8min.mat', './mat/0333_8min.mat',
		'./mat/0332_8min.mat', './mat/0122_8min.mat', './mat/0123_8min.mat', './mat/0009_8min.mat', './mat/0148_8min.mat', './mat/0149_8min.mat',
		'./mat/0311_8min.mat', './mat/0142_8min.mat', './mat/0325_8min.mat', './mat/0134_8min.mat', './mat/0150_8min.mat', './mat/0127_8min.mat',
		'./mat/0309_8min.mat', './mat/0147_8min.mat', './mat/0038_8min.mat', './mat/0105_8min.mat', './mat/0104_8min.mat', './mat/0032_8min.mat',
		'./mat/0103_8min.mat', './mat/0035_8min.mat', './mat/0313_8min.mat', './mat/0312_8min.mat', './mat/0016_8min.mat', './mat/0330_8min.mat',
		'./mat/0331_8min.mat', './mat/0121_8min.mat', './mat/0029_8min.mat', './mat/0028_8min.mat', './mat/0115_8min.mat', './mat/0023_8min.mat'
		]

	for i in range(len(capnobase_files)):
		extract_data(capnobase_files[i], f'./csv/capnobase_{capnobase_files[i][6:10]}.csv')

if __name__ == "__main__":
	main()