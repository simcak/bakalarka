from bcpackage import globals as G, show
from bcpackage.capnopackage import cb_data
from bcpackage.butpackage import but_data

from CapnoBase.cb_main import capnobase_main
from CapnoBase.cb_short_main import capnobase_main_short
from BUT_PPG.but_main import but_ppg_main

def main():
	G.CB_FILES, G.CB_FILES_LEN = cb_data.info()
	G.BUT_DATA_LEN = but_data.info()

	capnobase_main_short('my', first=True)
	capnobase_main('my')
	capnobase_main('neurokit')
	but_ppg_main('my')
	but_ppg_main('neurokit')

	show.full_results()

if __name__ == "__main__":
	main()