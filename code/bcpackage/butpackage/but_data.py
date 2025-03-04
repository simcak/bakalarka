import scipy.io
import pandas as pd
import csv
import scipy.io
import wfdb

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
	def _reader_general(file):
		"""
		Read the csv file.
		"""
		with open(f'./BUT_PPG/databases/big/{file}', 'r', encoding='utf-8-sig') as csvfile:
			reader = csv.DictReader(csvfile)
			data = list(reader)
		return data

	def _export_data_big(but_signal_info, i):
		"""
		Export the data into a CSV file.
		"""
		# Prepare data for CSV = Create a DataFrame
		row = []
		row.append(but_signal_info)
		data_row = pd.DataFrame(row)

		# Save DataFrame to CSV
		if (i == 0):
			with open('./BUT_PPG/databases/BUT_PPG_big.csv', 'w', newline='') as csvfile:
				data_row.to_csv(csvfile, header=True, index=False)
		else:
			with open('./BUT_PPG/databases/BUT_PPG_big.csv', 'a', newline='') as csvfile:
				data_row.to_csv(csvfile, header=False, index=False)


	# Load the CSV file
	subject_info = _reader_general('subject-info.csv')
	quality_hr = _reader_general('quality-hr-ann.csv')

	# Extract individual components
	row_subject_info = subject_info[i]
	row_quality_hr	 = quality_hr[i]
	# Extract specially ID for accurate file approaching
	id = row_subject_info['ID']

	# Read the PPG signal (data + header)
	record = wfdb.rdrecord('./BUT_PPG/databases/big/' + id + '/' + id + '_PPG')
	ppg_signal = record.p_signal
	ppg_fs = record.fs

	# Load the annotation - QRS - file. Extract the qrs and under-sample it from 1000 to 30 fs
	qrs_annot = wfdb.rdann('./BUT_PPG/databases/big/' + id + '/' + id, 'qrs')
	qrs_positions = qrs_annot.sample
	resampled_qrs_positions = [int(i * ppg_fs / 1000) for i in qrs_positions]

	but_signal_info = {
		'ID':			id,
		'ID-record':	f'{id[:3]}-{id[3:]}',
		'PPG_fs':		ppg_fs,
		'Gender':		row_subject_info['Gender'],
		'Age':			row_subject_info['Age [years]'],
		'Height':		row_subject_info['Height [cm]'],
		'Weight':		row_subject_info['Weight [kg]'],
		'Finger/Ear':	row_subject_info['Ear/finger'],
		'Motion':		row_subject_info['Motion'],
		'Quality':		row_quality_hr['Quality'],
		'HR':			row_quality_hr['HR'],
		'QRS':			resampled_qrs_positions,
		# 'PPG_Signal':	ppg_signal.flatten()
	}

	if export:
		_export_data_big(but_signal_info, i)

	return but_signal_info
