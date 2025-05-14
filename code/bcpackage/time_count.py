import time
import sys
import threading

def terminal_time():
	"""
	Print the time elapsed since the last call of this function.
	"""
	start_time = time.time()

	# Loading icon
	def print_loading_icon():
		icons = ['|', '/', '-', '\\']
		while not stop_event.is_set():
			for icon in icons:
				sys.stdout.write(f'\rProcessing {icon}')
				sys.stdout.flush()
				time.sleep(0.1)

	stop_event = threading.Event()
	loading_thread = threading.Thread(target=print_loading_icon)
	loading_thread.daemon = True
	loading_thread.start()

	return start_time, stop_event

def stop_terminal_time(start_time, stop_event, func_name='function'):
	"""
	Stop the loading icon and print the elapsed time.
	"""
	end_time = time.time()
	elapsed_time = end_time - start_time
	print(f"\n\033[96mTime spent in {func_name}: {elapsed_time:.2f} seconds\033[0m")
	stop_event.set()
