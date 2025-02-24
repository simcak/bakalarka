from . import globals as G

import pandas as pd
import numpy as np
import csv

###################################################################################
def to_csv_local(id, chunk_idx, i, hr_info, quality_info,
				 tp, fp, fn, sensitivity, precision,
				 type='My', database='CB', first=False):
	"""
	Framework for exporting chosen data and results of one signal into a CSV file.
	
	precision: Positive Predictivity
	"""
	# Prepare data for CSV
	rows = []
	if (database == 'CB'):
		if (type == 'My'):
			rows.append({
				'ID': f'{id}_{chunk_idx}min',
				'Sensitivity': sensitivity, 'Precision (PPV)': precision,
				'Our Quality': quality_info['Calc Q.'],
				'Diff HR[bpm]': hr_info['Diff HR'],
				'TP': tp, 'FP': fp, 'FN': fn
			})
		elif (type == 'NK'):
			rows.append({
				'ID': f'{id}_{chunk_idx}min',
				'Sensitivity': sensitivity, 'Precision (PPV)': precision,
				f'Orph. Q. (>={G.CORRELATION_THRESHOLD} = 1)': quality_info['Calc Q.'],
				'Diff HR[bpm]': hr_info['Diff HR'],
				'TP': tp, 'FP': fp, 'FN': fn
			})
	elif (database == 'BUT'):
		if (type == 'My'):
			rows.append({
				'ID': id,
				'Diff HR[bpm]': hr_info['Diff HR'],
				'Ref. Quality': quality_info['Ref Q.'],
				f'Our Q. (>={G.MORPHO_THRESHOLD})': quality_info['Calc Q.'],
				'Diff Quality': quality_info['Diff Q.']
			})
		elif (type == 'NK'):
			rows.append({
				'ID': id,
				'Diff HR[bpm]': hr_info['Diff HR'],
				'Ref. Quality': quality_info['Ref Q.'],
				f'Orph. Q. (>={G.CORRELATION_THRESHOLD})': quality_info['Calc Q.'],
				'Diff Quality': quality_info['Diff Q.']
			})
	else:
		raise ValueError("Invalid type provided for local export.")

	# Create a DataFrame
	data_row = pd.DataFrame(rows)

	# Append DataFrame to CSV
	if (i == 0 and chunk_idx == 0 and first == True):
		with open('./results.csv', 'w', newline='') as csvfile:
			csv.writer(csvfile).writerow([f'{database} {type}'])
			data_row.to_csv(csvfile, header=True, index=False)
	elif (i == 0 and first == False):
		with open('./results.csv', 'a', newline='') as csvfile:
			csv.writer(csvfile).writerow([])
			csv.writer(csvfile).writerow([f'{database} {type}'])
			data_row.to_csv(csvfile, header=True, index=False)
	# All data rows AFTER the 1st one
	else:
		with open('./results.csv', 'a', newline='') as csvfile:
			data_row.to_csv(csvfile, header=False, index=False)

###################################################################################
def to_csv_global(id,
				  sensitivity, precision,
				  type='My', database='CB'):
	"""
	Framework for exporting chosen data and results of the entire database into
	a CSV file.
	It is used for the final results.

	precision: Positive Predictivity
	"""
	row = []
	if (database == 'CB'):
		if (type == 'My'):
			row.append({
				'ID': id,
				'Total Se': sensitivity, 'Total PPV': precision,
				'AVG Quality': np.average(G.QUALITY_LIST), 'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		elif (type == 'NK'):
			row.append({
				'ID': id,
				'Total Se': sensitivity, 'Total PPV': precision,
				'AVG Quality': np.average(G.QUALITY_LIST), 'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	elif (database == 'BUT'):
		if (type == 'My'):
			row.append({
				'ID': id,
				'AVG Diff HR': np.average(G.DIFF_HR_LIST), 'AVG Diff Q-HR': np.average(G.DIFF_HR_LIST_QUALITY), 'Diff Quality': f'{G.DIFF_QUALITY_SUM} ({np.round(G.DIFF_QUALITY_SUM/G.BUT_DATA_LEN * 100, 3)}%)'
			})
		elif (type == 'NK'):
			row.append({
				'ID': id,
				'AVG Diff HR': np.average(G.DIFF_HR_LIST), 'AVG Diff Q-HR': np.average(G.DIFF_HR_LIST_QUALITY), 'Diff Quality': f'{G.DIFF_QUALITY_SUM} ({np.round(G.DIFF_QUALITY_SUM/G.BUT_DATA_LEN * 100, 3)}%)'
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	else:
		raise ValueError("Invalid databaze provided for global export.")

	global_data = pd.DataFrame(row)

	with open('./results.csv', 'a', newline='') as csvfile:
		csv.writer(csvfile).writerow([])
		global_data.to_csv(csvfile, header=True, index=False)
