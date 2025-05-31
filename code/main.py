from bcpackage import globals as G, show
from bcpackage.capnopackage import cb_data
from bcpackage.butpackage import but_data

from CapnoBase.cb_main import capnobase_main
from BUT_PPG.but_main import but_ppg_main

def main():
	############# CapnoBase Database #############
	##############################################
	G.CB_FILES, G.CB_FILES_LEN = cb_data.info()
	# capnobase_main('my', chunk=True, first=True)
	# capnobase_main('my')
	# capnobase_main('neurokit', chunk=True)
	# capnobase_main('neurokit')

	################ BUT Database ################
	##############################################
	G.BUT_DATA_LEN = but_data.info()
	# but_ppg_main('my')
	# but_ppg_main('neurokit')

	############## HJORTH Algorithm ##############
	from bcpackage import hjorth
	import pandas as pd
	import numpy as np

	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=48, autocorr_iterations=7, compute_quality=True)
	# hjorth.hjorth_alg(database='BUT_PPG', autocorr_iterations=7, compute_quality=False)

	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=1, autocorr_iterations=7)
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=8, autocorr_iterations=7, compute_quality=False)

	##### Quality Algorithm - HJORTH #####
	# hjorth.quality_hjorth_rf(database='all', print_show=False)

	# our_quality = hjorth.quality_hjorth(database='all', find_best_parameters=False)
	# file = pd.read_csv('hjorth_butppg.csv')
	# file["ourQ_this_only"] = our_quality
	# file.to_csv('hjorth_butppg.csv', index=False)

	# hjorth.hjorth_show_hr_all(database='CapnoBaseFull300')
	# hjorth.hjorth_show_only_quality_hr(database='BUT_PPG')
	# hjorth.hjorth_show_spi(database='CapnoBase300')

	############### Orphanidou Quality Algorithm ###############
	############################################################
	from bcpackage import quality
	chunked_pieces = 48
	# quality.ref_quality_orphanidou(database='CapnoBase', chunked_pieces=chunked_pieces)
	# f1_arr = []
	# for thr in [i * 0.01 for i in range(0, 101)]:
	# 	f1_arr.append((thr, quality.orphanidou_quality_evaluation(thr)))
	# max_f1 = max(f1_arr, key=lambda x: x[1])

	# threshold = max_f1[0]
	# quality.orphanidou_quality_evaluation(threshold, print_out=True)
	threshold = -0.1
	# quality.orphanidou_quality_plot(threshold, chunked_pieces)

	############## Show the results ##############
	##############################################
	# tables = show.full_results()
	# table_dict = {table['title']: table for table in tables}
	# if 'CB My chunked' in table_dict and 'CB NK chunked' in table_dict:
	# 	show.plotting_SePPV(table_dict['CB My chunked'], table_dict['CB NK chunked'], chunked=True)
	# if 'CB My all' in table_dict and 'CB NK all' in table_dict:
	# 	show.plotting_SePPV(table_dict['CB My all'], table_dict['CB NK all'])

	def show_bland_altman():
		# if 'CB My all' in table_dict:
		# 	table = table_dict['CB My all']
		# 	_id, reference, prediction = table['df']['ID'], table['df']['Ref HR[bpm]'], table['df']['Calculated HR[bpm]']
		# 	show.plot_bland_altman(_id, reference, prediction, title='Vlastní vrcholová detekce - celá CapnoBase', full=True)
		# if 'CB My chunked' in table_dict:
		# 	table = table_dict['CB My chunked']
		# 	_id, reference, prediction = table['df']['ID'], table['df']['Ref HR[bpm]'], table['df']['Calculated HR[bpm]']
		# 	show.plot_bland_altman(_id, reference, prediction, title='Vlastní vrcholová detekce - rozdělená CapnoBase')
		# if 'CB NK all' in table_dict:
		# 	table = table_dict['CB NK all']
		# 	_id, reference, prediction = table['df']['ID'], table['df']['Ref HR[bpm]'], table['df']['Calculated HR[bpm]']
		# 	show.plot_bland_altman(_id, reference, prediction, title='Elgendi - celá CapnoBase', full=True)
		# if 'CB NK chunked' in table_dict:
		# 	table = table_dict['CB NK chunked']
		# 	_id, reference, prediction = table['df']['ID'], table['df']['Ref HR[bpm]'], table['df']['Calculated HR[bpm]']
		# 	show.plot_bland_altman(_id, reference, prediction, title='Elgendi - rozdělená CapnoBase')
		# file = pd.read_csv('hjorth_CB_full_300.csv')
		# _id, reference, prediction = file['File name'].values, file['Ref HR'].values, file['Hjorth HR'].values
		# show.plot_bland_altman(_id, reference, prediction, title='Hjorth - celá CapnoBase', full=True)
		file = pd.read_csv('hjorth_CB_300.csv')
		file = file[file['Orphanidou Quality'] > 0.9]
		_id, reference, prediction = file['File name'].values, file['Ref HR'].values, file['Hjorth HR'].values
		show.plot_bland_altman(_id, reference, prediction, title='Hjorth - rozdělená CapnoBase')

		# if 'BUT My all' in table_dict:
		# 	show.plot_bland_altman(table_dict['BUT My all'])
		# if 'BUT NK all' in table_dict:
		# 	show.plot_bland_altman(table_dict['BUT NK all'])
		# if 'BUT My all' in table_dict:
		# 	show.plot_bland_altman(table_dict['BUT My all'])

	# if 'BUT My all' in table_dict and 'BUT NK all' in table_dict:
	# 	show.plotting_hr_diffs(table_dict['BUT My all'], table_dict['BUT NK all'])
	def but_ppg_table_results(table, title):
		_id = table['ID'].values
		ref_hr = table['Ref HR[bpm]'].values
		calc_hr = table['Calculated HR[bpm]'].values
		diff_hr = table['Diff HR[bpm]'].values
		# sdnn = table['SDNN [s]'].values
		# rmssd = table['RMSSD [s]'].values
		ref_q = table['Ref Quality'].values
		orphanidou_q = table['Orphanidou Quality'].values

		print("\nAll signals:")
		mea = np.mean(diff_hr)
		print(f"MAE: {mea:.2f} bpm")
		good_hr, bad_hr = np.sum(diff_hr < 5), np.sum(diff_hr >= 5)
		print(f"Good HR: {good_hr}, Bad HR: {bad_hr}")

		print("\nR-SQI:")
		q1_mea = np.mean(diff_hr[ref_q == 1])
		print(f"MAE: {q1_mea:.2f} bpm")
		q1_good_hr, q1_bad_hr = np.sum(diff_hr[ref_q == 1] < 5), np.sum(diff_hr[ref_q == 1] >= 5)
		print(f"Good HR: {q1_good_hr}, Bad HR: {q1_bad_hr}")
		q1_id = _id[ref_q == 1]
		q1_ref_hr = ref_hr[ref_q == 1]
		q1_calc_hr = calc_hr[ref_q == 1]

		print("\nO-SQI:")
		q2_mea = np.mean(diff_hr[orphanidou_q >= 0.9])
		print(f"MAE: {q2_mea:.2f} bpm")
		q2_good_hr, q2_bad_hr = np.sum(diff_hr[orphanidou_q >= 0.9] < 5), np.sum(diff_hr[orphanidou_q >= 0.9] >= 5)
		print(f"Good HR: {q2_good_hr}, Bad HR: {q2_bad_hr}")
		q2_id = _id[orphanidou_q >= 0.9]
		q2_ref_hr = ref_hr[orphanidou_q >= 0.9]
		q2_calc_hr = calc_hr[orphanidou_q >= 0.9]

		show.plot_bland_altman(_id, ref_hr, calc_hr, title=f'{title} celá databáze')
		show.plot_bland_altman(q1_id, q1_ref_hr, q1_calc_hr, title=f'{title} R-SQI')
		show.plot_bland_altman(q2_id, q2_ref_hr, q2_calc_hr, title=f'{title} O-SQI')

	# print("\n\033[96mBUT PPG - Our peak detection results:\033[0m")
	# table = pd.read_csv('results_butppg_our.csv')
	# but_ppg_table_results(table, "Vlastní vrcholová detekce -")

	# print("\n\033[96mBUT PPG - Elgendi peak detection results:\033[0m")
	# table = pd.read_csv('results_butppg_elgendi.csv')
	# but_ppg_table_results(table, "Elgendi -")

	# print("\n\033[96mBUT PPG - Hjorth peak detection results:\033[0m")
	# table = pd.read_csv('hjorth_butppg.csv')
	# but_ppg_table_results(table, "Hjorth -")


if __name__ == "__main__":
	main()