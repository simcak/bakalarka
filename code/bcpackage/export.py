import pandas as pd
import csv

def to_csv_local(id, tp, fp, fn, sensitivity, positive_predictivity, ref_hr, our_hr, diff_hr, i):
	"""
	Framework for exporting chosen data and results of one signal into a CSV file.
	"""
	# Prepare data for CSV
	rows = []
	rows.append({
		'ID': id,
		'TP': tp, 'FP': fp, 'FN': fn,
		'Sensitivity': sensitivity, 'Positive Predictivity': positive_predictivity,
		'Ref HR[bpm]' : ref_hr, 'Our HR[bpm]': our_hr, 'Diff HR[bpm]': diff_hr
	})

	# Create a DataFrame
	data_row = pd.DataFrame(rows)

	# Append DataFrame to CSV
	if (i == 0):
		with open('./results.csv', 'w', newline='') as csvfile:
			data_row.to_csv(csvfile, header=True, index=False)
	with open('./results.csv', 'a', newline='') as csvfile:
		data_row.to_csv(csvfile, header=False, index=False)

def to_csv_global(id, tp, fp, fn, sensitivity, positive_predictivity):
	"""
	Framework for exporting chosen data and results of the entire database into a CSV file.
	"""
	row = []
	row.append({
		'ID': id,
		'TP': tp, 'FP': fp, 'FN': fn,
		'Sensitivity': sensitivity, 'Positive Predictivity': positive_predictivity
	})

	global_data = pd.DataFrame(row)

	with open('./results.csv', 'a', newline='') as csvfile:
		csv.writer(csvfile).writerow([])
		global_data.to_csv(csvfile, header=True, index=False)
		
