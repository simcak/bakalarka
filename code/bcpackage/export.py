from . import globals as G

import pandas as pd
import numpy as np
import csv

###################################################################################
def to_csv_local(id_, chunk_idx, i, hr_info, statistical_info,
				 type='My', database='CB', first=False):
	"""
	Framework for exporting chosen data and results of one signal into a CSV file.
	
	precision: Positive Predictivity
	"""
	# Prepare data for CSV
	rows = []
	if (database == 'CB'):
		# assign
		tp, fp, fn = statistical_info['TP'], statistical_info['FP'], statistical_info['FN']
		sensitivity, precision, f1 = statistical_info['Se'], statistical_info['PPV'], statistical_info['F1']

		if (type == 'My'):
			rows.append({
				'ID': f'{id_}_{chunk_idx}min',
				'Sensitivity': sensitivity * 100, 'Precision (PPV)': precision * 100, 'F1': f1 * 100,
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
				'TP': tp, 'FP': fp, 'FN': fn
			})
		elif (type == 'NK'):
			rows.append({
				'ID': f'{id_}_{chunk_idx}min',
				'Sensitivity': sensitivity * 100, 'Precision (PPV)': precision * 100, 'F1': f1 * 100,
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
				'TP': tp, 'FP': fp, 'FN': fn
			})
	elif (database == 'BUT'):
		if (type == 'My'):
			rows.append({
				'ID': f'{id_[:3]}-{id_[3:]}',
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
			})
		elif (type == 'NK'):
			rows.append({
				'ID': f'{id_[:3]}-{id_[3:]}',
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
			})
	else:
		raise ValueError("Invalid type provided for local export.")

	# Create a DataFrame
	data_row = pd.DataFrame(rows)

	# Append DataFrame to CSV
	if (i == 0 and (chunk_idx == 0 or chunk_idx == 8) and first == True):
		with open('./results.csv', 'w', newline='') as csvfile:
			if (chunk_idx == 0):
				csv.writer(csvfile).writerow([f'{database} {type} chunked:'])
			if (chunk_idx == 8):
				csv.writer(csvfile).writerow([f'{database} {type} all:'])
			data_row.to_csv(csvfile, header=True, index=False)
	elif (i == 0 and (chunk_idx == 0 or chunk_idx == 8) and first == False):
		with open('./results.csv', 'a', newline='') as csvfile:
			csv.writer(csvfile).writerow([])
			if (chunk_idx == 0):
				csv.writer(csvfile).writerow([f'{database} {type} chunked:'])
			if (chunk_idx == 8):
				csv.writer(csvfile).writerow([f'{database} {type} all:'])
			data_row.to_csv(csvfile, header=True, index=False)
	# All data rows AFTER the 1st one
	else:
		with open('./results.csv', 'a', newline='') as csvfile:
			data_row.to_csv(csvfile, header=False, index=False)

###################################################################################
def to_csv_global(sensitivity, precision, type='My', database='CB'):
	"""
	Framework for exporting chosen data and results of the entire database into
	a CSV file.
	It is used for the final results.

	precision: Positive Predictivity
	"""
	row = []
	if (database == 'CB'):
		f1_total = 2 * (sensitivity * precision) / (sensitivity + precision)
		if (type == 'My'):
			row.append({
				'Total Se': sensitivity * 100, 'Total PPV': precision * 100, 'Total F1': f1_total * 100,
				'MAE': np.mean(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		elif (type == 'NK'):
			row.append({
				'Total Se': sensitivity * 100, 'Total PPV': precision * 100, 'Total F1': f1_total * 100,
				'MAE': np.mean(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	elif (database == 'BUT'):
		if (type == 'My'):
			row.append({
				'MAE': np.mean(G.DIFF_HR_LIST)
			})
		elif (type == 'NK'):
			row.append({
				'MAE': np.mean(G.DIFF_HR_LIST)
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	else:
		raise ValueError("Invalid databaze provided for global export.")

	global_data = pd.DataFrame(row)

	with open('./results.csv', 'a', newline='') as csvfile:
		csv.writer(csvfile).writerow([])
		csv.writer(csvfile).writerow([f'{database} {type} global:'])
		global_data.to_csv(csvfile, header=True, index=False)
