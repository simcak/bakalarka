from CapnoBase.cb_main import capnobase_main
from CapnoBase.cb_elgendi import capnobase_elgendi
from BUT_PPG.but_main import but_ppg_main

def main():
	capnobase_main()
	but_ppg_main()
	capnobase_elgendi()

if __name__ == "__main__":
	main()