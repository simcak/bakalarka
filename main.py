import scipy.io
import numpy as np
import pandas as pd
import h5py


def print_h5_structure(name, obj):
	"""Helper function to print out HDF5 structure recursively."""
	if isinstance(obj, h5py.Dataset):
		# If it's a dataset, print the name and the data
		print(f"Dataset: {name}")
		print(f"Data: {obj[()]}\n")
	elif isinstance(obj, h5py.Group):
		# If it's a group, print its name
		print(f"Group: {name}")


def main():
	# Load the .mat file
	mat_data = scipy.io.loadmat('./databaze/BUT_PPG.mat')

	# Access the main key containing the data
	data_key = 'BUT_PPG'
	structured_array = mat_data[data_key]

	# Extract individual components
	ppg_data = structured_array['PPG'][0, 0]  # PPG signals (2D array)
	ppg_fs = structured_array['PPG_fs'][0, 0]  # Sampling frequency
	ids = structured_array['ID'][0, 0]  # IDs
	quality = structured_array['Quality'][0, 0]  # Quality labels
	hr = structured_array['HR'][0, 0]  # Heart rates

	# Flatten arrays where needed
	ids_arr = ids.flatten()
	quality_arr = quality.flatten()
	hr_arr = hr.flatten()
	ppg_fs_value = int(ppg_fs[0])  # Assuming all have the same sampling frequency

	# Prepare data for CSV
	rows = []
	for i, ppg_signal in enumerate(ppg_data):
		rows.append({
			'ID': ids_arr[i],
			'PPG_fs': ppg_fs_value,  # Use the same sampling frequency for all
			'Quality': quality_arr[i],
			'HR': hr_arr[i],
			'PPG_Signal': list(ppg_signal)  # Convert array to list for easier storage
		})

	# Create a DataFrame
	df = pd.DataFrame(rows)

	# Save DataFrame to CSV
	df.to_csv('but_ppg_dataset.csv', index=False)



	# Specify the path to the .mat file
	capnobase_path = './databaze/capnobase/0009_8min.mat'

	# Load the .mat file using h5py
	with h5py.File(capnobase_path, 'r') as mat_data:
		# Use the visititems method to go through each item in the HDF5 file
		mat_data.visititems(print_h5_structure)

if __name__ == "__main__":
	main()