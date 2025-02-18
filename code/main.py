from CapnoBase.cb_main import capnobase_main
from BUT_PPG.but_main import but_ppg_main
from bcpackage import constants as C
from bcpackage.capnopackage import cb_data
from bcpackage.butpackage import but_data

def main():
	C.CB_FILES, C.CB_FILES_LEN = cb_data.info()
	C.BUT_DATA_LEN = but_data.info()

	capnobase_main('my')
	capnobase_main('neurokit')
	but_ppg_main('my')
	but_ppg_main('neurokit')

if __name__ == "__main__":
	main()