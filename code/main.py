from bcpackage import globals as G, show
from bcpackage.capnopackage import cb_data
from bcpackage.butpackage import but_data

from CapnoBase.cb_main import capnobase_main
from BUT_PPG.but_main import but_ppg_main

def main():
	G.CB_FILES, G.CB_FILES_LEN = cb_data.info()
	G.BUT_DATA_LEN = but_data.info()

	but_ppg_main('my')

	# capnobase_main('my', chunk=True, first=True)
	# capnobase_main('neurokit', chunk=True)
	# capnobase_main('my')
	# capnobase_main('neurokit')
	# but_ppg_main('my')
	# but_ppg_main('neurokit')

	tables = show.full_results()
	show.plotting_SePPV(tables[0], tables[2], chunked=True)
	show.plotting_SePPV(tables[4], tables[6])

if __name__ == "__main__":
	main()