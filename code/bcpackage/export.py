from . import globals as G

import pandas as pd
import numpy as np
import csv

###################################################################################
def to_csv_local(id_, chunk_idx, i, hr_info, quality_info, statistical_info,
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
		sensitivity, precision = statistical_info['Se'], statistical_info['PPV']

		if (type == 'My'):
			rows.append({
				'ID': f'{id_}_{chunk_idx}min',
				'Sensitivity': sensitivity * 100, 'Precision (PPV)': precision * 100,
				'Our Quality': quality_info['Calc Q.'] * 100,
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
				'TP': tp, 'FP': fp, 'FN': fn
			})
		elif (type == 'NK'):
			rows.append({
				'ID': f'{id_}_{chunk_idx}min',
				'Sensitivity': sensitivity * 100, 'Precision (PPV)': precision * 100,
				f'Orph. Q. (>={G.CORRELATION_THRESHOLD * 100}%)': quality_info['Calc Q.'] * 100,
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
				'Ref. Quality': quality_info['Ref Q.'],
				f'Our Q. (>={G.MORPHO_THRESHOLD * 100}%)': quality_info['Calc Q.'] * 100,
				'Diff Quality': quality_info['Diff Q.']
			})
		elif (type == 'NK'):
			rows.append({
				'ID': f'{id_[:3]}-{id_[3:]}',
				'Diff HR[bpm]': hr_info['Diff HR'],
				'SDNN [s]': hr_info['SDNN'], 'RMSSD [s]': hr_info['RMSSD'],
				'Ref. Quality': quality_info['Ref Q.'],
				f'Orph. Q. (>={G.CORRELATION_THRESHOLD * 100}%)': quality_info['Calc Q.'] * 100,
				'Diff Quality': quality_info['Diff Q.']
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
		if (type == 'My'):
			row.append({
				'Total Se': sensitivity * 100, 'Total PPV': precision * 100,
				'AVG Quality': np.average(G.QUALITY_LIST) * 100, 'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		elif (type == 'NK'):
			row.append({
				'Total Se': sensitivity * 100, 'Total PPV': precision * 100,
				'AVG Quality': np.average(G.QUALITY_LIST) * 100, 'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'TP sum': np.sum(G.TP_LIST), 'FP sum': np.sum(G.FP_LIST), 'FN sum': np.sum(G.FN_LIST)
			})
		else:
			raise ValueError("Invalid type provided for global export.")
	elif (database == 'BUT'):
		if (type == 'My'):
			row.append({
				'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'AVG Diff Q-HR': np.average(G.DIFF_HR_LIST_QUALITY),
				'Diff Quality': f'{G.DIFF_QUALITY_SUM} ({np.round(G.DIFF_QUALITY_SUM/G.BUT_DATA_LEN * 100, 3)}%)'
			})
		elif (type == 'NK'):
			row.append({
				'AVG Diff HR': np.average(G.DIFF_HR_LIST),
				'AVG Diff Q-HR': np.average(G.DIFF_HR_LIST_QUALITY),
				'Diff Quality': f'{G.DIFF_QUALITY_SUM} ({np.round(G.DIFF_QUALITY_SUM/G.BUT_DATA_LEN * 100, 3)}%)'
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
