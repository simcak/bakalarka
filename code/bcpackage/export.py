import pandas as pd
import csv

def to_csv_local(id, tp, fp, fn, sensitivity, precision, ref_hr, our_hr, diff_hr, i, type='capnobase'):
	"""
	Framework for exporting chosen data and results of one signal into a CSV file.
	
	precision: Positive Predictivity
	"""
	# Prepare data for CSV
	rows = []
	if (type == 'capnobase'):
		rows.append({
			'ID': id,
			'TP': tp, 'FP': fp, 'FN': fn,
			'Sensitivity': sensitivity, 'Precision (PPV)': precision,
			'Ref HR[bpm]' : ref_hr, 'Our HR[bpm]': our_hr, 'Diff HR[bpm]': diff_hr
		})
	if (type == 'but_ppg'):
		rows.append({
			'ID': id,
			'TP': None, 'FP': None, 'FN': None,
			'Sensitivity': sensitivity, 'Precision (PPV)': precision,
			'Ref HR[bpm]' : ref_hr, 'Our HR[bpm]': our_hr, 'Diff HR[bpm]': diff_hr,
			'Quality': None
		})

	# Create a DataFrame
	data_row = pd.DataFrame(rows)

	# Append DataFrame to CSV
	if (i == 0 and type == 'capnobase'):
		with open('./results.csv', 'w', newline='') as csvfile:
			data_row.to_csv(csvfile, header=True, index=False)
	if (i == 0 and type == 'but_ppg'):
		with open('./results.csv', 'a', newline='') as csvfile:
			csv.writer(csvfile).writerow([])
			data_row.to_csv(csvfile, header=True, index=False)
	with open('./results.csv', 'a', newline='') as csvfile:
		data_row.to_csv(csvfile, header=False, index=False)

def to_csv_global(id, tp, fp, fn, sensitivity, precision, diff_hr):
	"""
	Framework for exporting chosen data and results of the entire database into a CSV file.
	It is used for the final results.

	precision: Positive Predictivity
	"""
	row = []
	row.append({
		'ID': id,
		'TP': tp, 'FP': fp, 'FN': fn,
		'Sensitivity': sensitivity, 'Precision (PPV)': precision,
		'Empty_1' : None, 'Empty_2': None,
		'Average Diff HR[bpm]': diff_hr
	})

	global_data = pd.DataFrame(row)

	with open('./results.csv', 'a', newline='') as csvfile:
		csv.writer(csvfile).writerow([])
		global_data.to_csv(csvfile, header=True, index=False)
		
