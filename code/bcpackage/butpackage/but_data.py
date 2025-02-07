import scipy.io
import pandas as pd

def export_data(id, fs, quality, hr, ppg_signal, i):
	# Prepare data for CSV
	row = []
	row.append({
		'ID': id,
		'PPG_fs': fs,
		'Quality': quality,
		'HR': hr,
		'PPG_Signal': list(ppg_signal)	# Convert array to list for easier storage in CSV
	})

	# Create a DataFrame
	data_row = pd.DataFrame(row)

	# Save DataFrame to CSV
	if (i == 0):
		with open('./but_database.csv', 'w', newline='') as csvfile:
			data_row.to_csv(csvfile, header=True, index=False)
	else:
		with open('./but_database.csv', 'a', newline='') as csvfile:
			data_row.to_csv(csvfile, header=False, index=False)

def extract(i, export=False):
	"""
	
	"""
	# Load the .mat file
	mat_data = scipy.io.loadmat('./BUT_PPG/BUT_PPG.mat')

	# Access the main key containing the data
	structured_array = mat_data['BUT_PPG']

	# Extract individual components
	id			= structured_array['ID'][0, 0].flatten()[i]
	fs			= int(structured_array['PPG_fs'][0, 0][0])
	quality		= structured_array['Quality'][0, 0].flatten()[i]
	hr			= structured_array['HR'][0, 0].flatten()[i]
	ppg_signal	= structured_array['PPG'][0, 0][i]

	if export:
		export_data(id, fs, quality, hr, ppg_signal, i)

	return id, fs, quality, hr, ppg_signal