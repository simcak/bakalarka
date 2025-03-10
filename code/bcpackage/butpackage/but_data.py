import scipy.io
import pandas as pd
import csv
import scipy.io
import wfdb

def info():
	"""
	Get the information about the BUT PPG database.
	"""
	with open('./BUT_PPG/databases/big/subject-info.csv', 'r', encoding='utf-8-sig') as csvfile:
		reader = csv.DictReader(csvfile)
		data = list(reader)

		return len(data)


def extract(i, export=False):
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

	def _export_data(but_signal_info, i):
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
	id_ = str(row_subject_info['ID'])

	# Read the PPG signal (data + header) - fs for export - we dont want to export signals to CSV (too big)
	record = wfdb.rdrecord('./BUT_PPG/databases/big/' + id_ + '/' + id_ + '_PPG')
	ppg_fs = int(record.fs)
	signal_data = record.p_signal
	signal_shape = signal_data.shape

	# Load the annotation - QRS - file. Extract the qrs and under-sample it from 1000 to 30 fs
	qrs_annot = wfdb.rdann('./BUT_PPG/databases/big/' + id_ + '/' + id_, 'qrs')
	qrs_positions = qrs_annot.sample
	resampled_qrs_positions = [int(i * ppg_fs / 1000) for i in qrs_positions]

	but_signal_info = {
		'ID':			id_,
		'ID-record':	f'{id_[:3]}-{id_[3:]}',
		'PPG_fs':		ppg_fs,
		'Gender':		row_subject_info['Gender'],
		'Age':			row_subject_info['Age [years]'],
		'Height':		row_subject_info['Height [cm]'],
		'Weight':		row_subject_info['Weight [kg]'],
		'Finger/Ear':	row_subject_info['Ear/finger'],
		'Motion':		row_subject_info['Motion'],
		'Ref_Quality':	int(row_quality_hr['Quality']),
		'Ref_HR':		int(row_quality_hr['HR']),
		'QRS':			resampled_qrs_positions
	}

	if export:
		_export_data(but_signal_info, i)

	# For the first, original, 48 signals, the signal is in the shape (1, 300)
	if signal_shape == (1, 300):
		but_signal_info['PPG_Red']		= None
		but_signal_info['PPG_Green']	= None
		but_signal_info['PPG_Blue']		= None
		but_signal_info['PPG_Signal']	= list(signal_data.flatten())
	# For the rest of the signals, the signal is in the shape (300, 3) = has RGB channels and we use only the Red one
	elif signal_shape == (300, 3):
		but_signal_info['PPG_Red']		= list(signal_data[:, 0])
		but_signal_info['PPG_Green']	= list(signal_data[:, 1])
		but_signal_info['PPG_Blue']		= list(signal_data[:, 2])
		but_signal_info['PPG_Signal']	= list(but_signal_info['PPG_Red'])

	return but_signal_info
