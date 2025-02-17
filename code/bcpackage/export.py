import pandas as pd
import csv

###################################################################################
def to_csv_local(id, i, ref_hr, our_hr, diff_hr,
				 tp, fp, fn, sensitivity, precision,
				 ref_quality, quality,
				 type='my', database='CB'):
	"""
	Framework for exporting chosen data and results of one signal into a CSV file.
	
	precision: Positive Predictivity
	"""
	# Prepare data for CSV
	rows = []
	if (database == 'CB'):
		if (type == 'my'):
			rows.append({
				'ID': id,
				'Sensitivity': sensitivity, 'Precision (PPV)': precision,
				'Diff HR[bpm]': diff_hr,
				'Our Quality': None,
				'TP': tp, 'FP': fp, 'FN': fn
			})
		elif (type == 'NK'):
			rows.append({
				'ID': id,
				'Sensitivity': sensitivity, 'Precision (PPV)': precision,
				'Diff HR[bpm]': diff_hr,
				'Orph. Quality': quality,
				'TP': tp, 'FP': fp, 'FN': fn
			})
	elif (database == 'BUT'):
		if (type == 'my'):
			rows.append({
				'ID': id,
				'Diff HR[bpm]': diff_hr,
				'Ref. Quality': ref_quality, 'Our Quality': None, 'Diff Quality': None
			})
		elif (type == 'NK'):
			rows.append({
				'ID': id,
				'Diff HR[bpm]': diff_hr,
				'Ref. Quality': ref_quality, 'Orph. Quality': quality, 'Diff Quality': None
			})
	else:
		raise ValueError("Invalid type provided for local export.")

	# Create a DataFrame
	data_row = pd.DataFrame(rows)

	# Append DataFrame to CSV
	if (i == 0 and type == 'my' and database == 'CB'):
		with open('./results.csv', 'w', newline='') as csvfile:
			csv.writer(csvfile).writerow([f'{database} {type}'])
			data_row.to_csv(csvfile, header=True, index=False)
	elif (i == 0):
		with open('./results.csv', 'a', newline='') as csvfile:
			csv.writer(csvfile).writerow([])
			csv.writer(csvfile).writerow([f'{database} {type}'])
			data_row.to_csv(csvfile, header=True, index=False)
	# All data rows AFTER the 1st one
	else:
		with open('./results.csv', 'a', newline='') as csvfile:
			data_row.to_csv(csvfile, header=False, index=False)

###################################################################################
def to_csv_global(id, diff_hr, diff_Q_hr,
				  tp, fp, fn, sensitivity, precision,
				  type='my', database='CB'):
	"""
	Framework for exporting chosen data and results of the entire database into
	a CSV file.
	It is used for the final results.

	precision: Positive Predictivity
	"""
	row = []
	if (database == 'CB'):
		if (type == 'my'):
			row.append({
				'ID': id,
				'Total Se': sensitivity, 'Total PPV': precision,
				'AVG Diff HR': diff_hr, 'AVG Quality': None,
				'TP sum': tp, 'FP sum': fp, 'FN sum': fn
			})
		elif (type == 'NK'):
			row.append({
				'ID': id,
				'Total Se': sensitivity, 'Total PPV': precision,
				'AVG Diff HR': diff_hr, 'AVG Diff Quality': None,
				'TP sum': tp, 'FP sum': fp, 'FN sum': fn
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	elif (database == 'BUT'):
		if (type == 'my'):
			row.append({
				'ID': id,
				'AVG Diff HR': diff_hr, 'AVG Diff Q-HR': diff_Q_hr, 'AVG Diff Quality': None
			})
		elif (type == 'NK'):
			row.append({
				'ID': id,
				'AVG Diff HR': diff_hr, 'AVG Diff Q-HR': diff_Q_hr, 'AVG Diff Quality': None
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	else:
		raise ValueError("Invalid databaze provided for global export.")

	global_data = pd.DataFrame(row)

	with open('./results.csv', 'a', newline='') as csvfile:
		csv.writer(csvfile).writerow([])
		global_data.to_csv(csvfile, header=True, index=False)
