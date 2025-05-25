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
	# capnobase_main('neurokit', chunk=True)
	# capnobase_main('my')
	# capnobase_main('neurokit')

	################ BUT Database ################
	##############################################
	G.BUT_DATA_LEN = but_data.info()
	# but_ppg_main('my')
	# but_ppg_main('neurokit')

	############## HJORTH Algorithm ##############
	from bcpackage import hjorth
	import pandas as pd
	"""
		chunked_pieces 1 == 8min is full signal
		chunked_pieces 20 == 24s - best results = 0,66 avg HR diff
		chunked_pieces 48 == 10s like in BUT PPG = 0,82 avg HR diff || 0,81 after eliminating by Orphanidu (22 samples removed)
	"""
	hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=48, autocorr_iterations=7)
	# hjorth.hjorth_alg(database='BUT_PPG', autocorr_iterations=7, compute_quality=True)
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=1, autocorr_iterations=7)
	# hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=30, autocorr_iterations=7)

	##### Quality Algorithm - HJORTH #####
	# hjorth.quality_hjorth()
	# our_quality = hjorth.quality_hjorth(find_best_parameters=True)
	# file = pd.read_csv('hjorth.csv')
	# file["ourQ_this_only"] = our_quality
	# file.to_csv('hjorth.csv', index=False)

	# hjorth.confusion_matrix(database='all')
	hjorth.hjorth_show_hr(48, database='current')
	# hjorth.hjorth_show_only_quality_hr(48, database='CapnoBase300')
	# hjorth.hjorth_show_spi()

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
	# tables = show.full_results()
	# table_dict = {table['title']: table for table in tables}
	# if 'CB My chunked' in table_dict and 'CB NK chunked' in table_dict:
	# 	show.plotting_SePPV(table_dict['CB My chunked'], table_dict['CB NK chunked'], chunked=True)
	# if 'CB My all' in table_dict and 'CB NK all' in table_dict:
	# 	show.plotting_SePPV(table_dict['CB My all'], table_dict['CB NK all'])
	# if 'CB My all' in table_dict and 'CB NK all' in table_dict:
	# 	show.plotting_SDNR(table_dict['CB My all'], table_dict['CB NK all'])
	# if 'BUT My all' in table_dict and 'BUT NK all' in table_dict:
	# 	show.plotting_hr_diffs(table_dict['BUT My all'], table_dict['BUT NK all'])

if __name__ == "__main__":
	main()