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
	hjorth.hjorth_alg(database='CapnoBase', chunked_pieces=4, show=True)
	# hjorth.hjorth_alg(database='BUT_PPG', show=True)

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