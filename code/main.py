from bcpackage import globals as G, show
from bcpackage.capnopackage import cb_data
from bcpackage.butpackage import but_data

from CapnoBase.cb_main import capnobase_main
from BUT_PPG.but_main import but_ppg_main

def main():
	############# CapnoBase Database #############
	##############################################
	G.CB_FILES, G.CB_FILES_LEN = cb_data.info()
	capnobase_main('my', chunk=True, first=True)
	capnobase_main('my')
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
	"""
		chunked_pieces 1 == 8min is full signal
		chunked_pieces 20 == 24s - best results = 0,66 avg HR diff
		chunked_pieces 48 == 10s like in BUT PPG = 0,82 avg HR diff || 0,81 after eliminating by Orphanidu (22 samples removed)
	"""
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=48, autocorr_iterations=7, compute_quality=False)
	# hjorth.hjorth_alg(database='BUT_PPG', autocorr_iterations=7, compute_quality=True)
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=1, autocorr_iterations=7)
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=8, autocorr_iterations=7, compute_quality=False)

	##### Quality Algorithm - HJORTH #####
	# hjorth.quality_hjorth(database='all', find_best_parameters=False)
	# our_quality = hjorth.quality_hjorth(database='all', find_best_parameters=False)
	# file = pd.read_csv('hjorth_butppg.csv')
	# file["ourQ_this_only"] = our_quality
	# file.to_csv('hjorth_butppg.csv', index=False)

	# hjorth.confusion_matrix(database='all')
	# hjorth.hjorth_show_hr_all(database='CapnoBaseFull300')
	# hjorth.hjorth_show_only_quality_hr(database='CapnoBase300')
	# hjorth.hjorth_show_spi(database='CapnoBase300')

	############## Quality Algorithm - Orphanidou ##############
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
	tables = show.full_results()
	table_dict = {table['title']: table for table in tables}
	if 'CB My chunked' in table_dict and 'CB NK chunked' in table_dict:
		show.plotting_SePPV(table_dict['CB My chunked'], table_dict['CB NK chunked'], chunked=True)
	if 'CB My all' in table_dict and 'CB NK all' in table_dict:
		show.plotting_SePPV(table_dict['CB My all'], table_dict['CB NK all'])

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

if __name__ == "__main__":
	main()