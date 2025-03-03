import scipy.io
import pandas as pd
import csv
import scipy.io

def info():
	mat_data = scipy.io.loadmat('./BUT_PPG/databases/BUT_PPG.mat')
	ppg_data_len = len(mat_data['BUT_PPG']['PPG'][0, 0])

	return ppg_data_len


def extract(i, export=False):
	"""
	Extact the data from the old=short BUT PPG database.
	"""
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
			with open('./BUT_PPG/databases/but_database.csv', 'w', newline='') as csvfile:
				data_row.to_csv(csvfile, header=True, index=False)
		else:
			with open('./BUT_PPG/databases/but_database.csv', 'a', newline='') as csvfile:
				data_row.to_csv(csvfile, header=False, index=False)

	# Load the .mat file
	mat_data = scipy.io.loadmat('./BUT_PPG/databases/BUT_PPG.mat')

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


def extract_big(i, export=False):
	"""
	Extact the data from the new=big BUT PPG database.
	"""
	def export_data_big(id, gender, age, height, weight, ear_finger, motion, i):
		"""
		Export the data into a CSV file.
		"""
		# Prepare data for CSV
		row = []
		row.append({
			'ID':			id,
			'Gender':		gender,
			'Age':			age,
			'Height':		height,
			'Weight':		weight,
			'Ear/Finger':	ear_finger,
			'Motion':		motion
		})

		# Create a DataFrame
		data_row = pd.DataFrame(row)

		# Save DataFrame to CSV
		if (i == 0):
			with open('./BUT_PPG/databases/BUT_PPG_big.csv', 'w', newline='') as csvfile:
				data_row.to_csv(csvfile, header=True, index=False)
		else:
			with open('./BUT_PPG/databases/BUT_PPG_big.csv', 'a', newline='') as csvfile:
				data_row.to_csv(csvfile, header=False, index=False)


	# Load the CSV file
	with open('./BUT_PPG/databases/big/subject-info.csv', 'r') as csvfile:
		reader = csv.DictReader(csvfile)
		rows = list(reader)

	# Extract individual components
	row				= rows[i]
	id				= row['ID']
	gender			= row['Gender']
	age				= row['Age [years]']
	height			= row['Height [cm]']
	weight			= row['Weight [kg]']
	ear_finger		= row['Ear/finger']
	motion			= row['Motion']

	if export:
		export_data_big(id, gender, age, height, weight, ear_finger, motion, i)
		pass

	return id, gender, age, height, weight, ear_finger, motion