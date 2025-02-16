from CapnoBase.cb_main import capnobase_main
from BUT_PPG.but_main import but_ppg_main

def main():
	capnobase_main('my')
	but_ppg_main('my')
	capnobase_main('elgendi')
	but_ppg_main('elgendi')

if __name__ == "__main__":
	main()