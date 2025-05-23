from . import globals as G
import pandas as pd
from io import StringIO
from matplotlib import pyplot as plt
import numpy as np

def plotting_SDNR(table1, table2):
	plt.figure()
	plt.scatter(table2['df']['SDNN [s]'], table1['df']['SDNN [s]'])
	plt.plot([0, max(table2['df']['SDNN [s]'])], [0, max(table1['df']['SDNN [s]'])], 'r--')	# diagonála
	plt.xlabel('Referenční SDNN')
	plt.ylabel('Naše SDNN')
	plt.title('Porovnání SDNN (CapnoBase)')
	plt.show()

	mean_sdnn = (table1['df']['SDNN [s]'] + table2['df']['SDNN [s]']) / 2
	diff_sdnn = table1['df']['SDNN [s]'] - table2['df']['SDNN [s]']

	plt.figure()
	plt.scatter(mean_sdnn, diff_sdnn)
	plt.axhline(np.mean(diff_sdnn), color='gray', linestyle='--')
	plt.axhline(np.mean(diff_sdnn) + 1.96 * np.std(diff_sdnn), color='red', linestyle='--')
	plt.axhline(np.mean(diff_sdnn) - 1.96 * np.std(diff_sdnn), color='red', linestyle='--')
	plt.xlabel('Průměr SDNN')
	plt.ylabel('Rozdíl SDNN (My - Ref)')
	plt.title('Bland-Altman SDNN')
	plt.show()

	abs_error_sdnn = np.abs(table1['df']['SDNN [s]'] - table2['df']['SDNN [s]'])
	abs_error_rmssd = np.abs(table1['df']['RMSSD [s]'] - table2['df']['RMSSD [s]'])

	plt.boxplot([abs_error_sdnn, abs_error_rmssd], labels=['SDNN', 'RMSSD'])
	plt.ylabel('Absolutní chyba')
	plt.title('Rozložení chyb (CapnoBase)')
	plt.show()


def plotting_SePPV(table1, table2, chunked=False):
	fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7))

	# name of the figure
	fig.suptitle(f"Porovnání naší a Elgendiho metody pro CapnoBase dababázi", fontsize=16)

	################ Plot Sensitivity ################
	ax1.set_title(f"Citlivost (Se)")
	ax1.plot(table1['df']['ID'], table1['df']['Sensitivity'], label=f'Naše metoda', color=G.CESA_BLUE)
	ax1.plot(table2['df']['ID'], table2['df']['Sensitivity'], label=f'Elgendiho metoda', color=G.BUT_RED)
	# Calculate and plot the average sensitivity
	avg_sensitivity_table1 = table1['df']['Sensitivity'].mean()
	avg_sensitivity_table2 = table2['df']['Sensitivity'].mean()
	ax1.axhline(avg_sensitivity_table1, color=G.CESA_BLUE, linestyle='--', label=f'Náš průměr: {avg_sensitivity_table1:.2f}')
	ax1.axhline(avg_sensitivity_table2, color=G.BUT_RED, linestyle='--', label=f'Elgendiho průměr: {avg_sensitivity_table2:.2f}')
	##################################################

	######################################## SET AXIS ########################################
	ax1.set_ylabel('Citlivost [%]')
	if chunked:
		# Show only every 8th label on the x-axis
		ax1.set_xticks(range(0, len(table1['df']['ID']), 8))
		ax1.set_xticklabels([label.split('_')[0] for i, label in enumerate(table1['df']['ID']) if i % 8 == 0])
		# Add small marks for the rest of the signals
		ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
		ax1.tick_params(axis='x', which='minor', length=4, color='gray')
	else:
		# Show only the number before the '_' in the ID on the x-axis
		ax1.set_xticks(range(len(table1['df']['ID'])))
		ax1.set_xticklabels([label.split('_')[0] for label in table1['df']['ID']])
	#############################################################################################

	ax1.margins(x=0, y=0.1)
	ax1.legend(fontsize='small')
	ax1.tick_params(axis='x', rotation=90)

	#################### Plot Positive Predictive Value (PPV) ####################
	ax2.set_title(f"Přesnost (PPV)")
	ax2.plot(table1['df']['ID'], table1['df']['Precision (PPV)'], label=f'Naše metoda', color=G.CESA_BLUE)
	ax2.plot(table2['df']['ID'], table2['df']['Precision (PPV)'], label=f'Elgendiho metoda', color=G.BUT_RED)
	# Calculate and plot the average precision
	avg_precision_table1 = table1['df']['Precision (PPV)'].mean()
	avg_precision_table2 = table2['df']['Precision (PPV)'].mean()
	ax2.axhline(avg_precision_table1, color=G.CESA_BLUE, linestyle='--', label=f'Náš průměr: {avg_precision_table1:.2f}')
	ax2.axhline(avg_precision_table2, color=G.BUT_RED, linestyle='--', label=f'Elgendiho průměr: {avg_precision_table2:.2f}')
	##############################################################################

	######################################## SET AXIS ########################################
	ax2.set_ylabel('Přesnost [%]')
	if chunked:
		# Show only every 8th label on the x-axis
		ax2.set_xticks(range(0, len(table1['df']['ID']), 8))
		ax2.set_xticklabels([label.split('_')[0] for i, label in enumerate(table1['df']['ID']) if i % 8 == 0])
		# Add small marks for the rest of the signals
		ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
		ax2.tick_params(axis='x', which='minor', length=4, color='gray')
		ax2.set_xlabel('ID (shown just for the 1st minute)')
	else:
		# Show only the number before the '_' in the ID on the x-axis
		ax2.set_xticks(range(len(table1['df']['ID'])))
		ax2.set_xticklabels([label.split('_')[0] for label in table1['df']['ID']])
		ax2.set_xlabel('ID')
	#############################################################################################

	ax2.margins(x=0, y=0.1)
	ax2.legend(fontsize='small')
	ax2.tick_params(axis='x', rotation=90)

	plt.tight_layout()
	plt.show()


def plotting_hr_diffs(table1, table2):
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(13, 9))
	# name of the figure
	fig.suptitle(f"Porovnání naší a Elgendiho metody pro kvalitní signály VUT PPG databáze", fontsize=16)

	table_len = len(table1['df']['ID'])
	first_tr = int(table_len/3)
	second_tr = int(2 * first_tr)

	####################### Plot Diff HR 1st part #######################
	ax1.set_title(f'Rozdíly naměřených TF')
	ax1.plot(table1['df']['ID'][:first_tr], table1['df']['Diff HR[bpm]'][:first_tr], label=f'Naše metoda', color=G.CESA_BLUE)
	ax1.plot(table2['df']['ID'][:first_tr], table2['df']['Diff HR[bpm]'][:first_tr], label=f'Elgendiho metoda', color=G.BUT_RED)
	# Calculate and plot the average Diff HR
	avg_table1 = table1['df']['Diff HR[bpm]'][:first_tr].mean()
	avg_table2 = table2['df']['Diff HR[bpm]'][:first_tr].mean()
	ax1.axhline(avg_table1, color=G.CESA_BLUE, linestyle='--', label=f'Náš průměr: {avg_table1:.2f}')
	ax1.axhline(avg_table2, color=G.BUT_RED, linestyle='--', label=f'Elgendiho průměr: {avg_table2:.2f}')
	#####################################################################
	######################################## SET AXIS ########################################
	ax1.set_ylabel('Rozdíl [tep/min]')
	# Show only every 8th label on the x-axis
	ax1.set_xticks(range(0, len(table1['df']['ID'][:first_tr]), 4))
	ax1.set_xticklabels([label for i, label in enumerate(table1['df']['ID'][:first_tr]) if i % 4 == 0])
	# Add small marks for the rest of the signals
	ax1.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax1.tick_params(axis='x', which='minor', length=4, color='gray')
	###########################################################################################
	ax1.margins(x=0, y=0.1)
	ax1.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
	ax1.tick_params(axis='x', rotation=90)

	####################### Plot Diff HR 2nd part #######################
	ax2.plot(table1['df']['ID'][first_tr:second_tr], table1['df']['Diff HR[bpm]'][first_tr:second_tr], label=f'Naše metoda', color=G.CESA_BLUE)
	ax2.plot(table2['df']['ID'][first_tr:second_tr], table2['df']['Diff HR[bpm]'][first_tr:second_tr], label=f'Elgendiho metoda', color=G.BUT_RED)
	# Calculate and plot the average Diff HR
	avg_table1b = table1['df']['Diff HR[bpm]'][first_tr:second_tr].mean()
	avg_table2b = table2['df']['Diff HR[bpm]'][first_tr:second_tr].mean()
	ax2.axhline(avg_table1b, color=G.CESA_BLUE, linestyle='--', label=f'Náš průměr: {avg_table1b:.2f}')
	ax2.axhline(avg_table2b, color=G.BUT_RED, linestyle='--', label=f'Elgendiho průměr: {avg_table2b:.2f}')
	#####################################################################
	######################################## SET AXIS ########################################
	ax2.set_ylabel('Rozdíl [tep/min]')
	# Show only every 8th label on the x-axis
	ax2.set_xticks(range(0, len(table1['df']['ID'][first_tr:second_tr]), 4))
	ax2.set_xticklabels([label for i, label in enumerate(table1['df']['ID'][first_tr:second_tr]) if i % 4 == 0])
	# Add small marks for the rest of the signals
	ax2.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax2.tick_params(axis='x', which='minor', length=4, color='gray')
	###########################################################################################
	ax2.margins(x=0, y=0.1)
	ax2.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
	ax2.tick_params(axis='x', rotation=90)

	####################### Plot Diff HR 3rd part #######################
	ax3.plot(table1['df']['ID'][second_tr:], table1['df']['Diff HR[bpm]'][second_tr:], label=f'Naše metoda', color=G.CESA_BLUE)
	ax3.plot(table2['df']['ID'][second_tr:], table2['df']['Diff HR[bpm]'][second_tr:], label=f'Elgendiho metoda', color=G.BUT_RED)
	# Calculate and plot the average Diff HR
	avg_table1c = table1['df']['Diff HR[bpm]'][second_tr:].mean()
	avg_table2c = table2['df']['Diff HR[bpm]'][second_tr:].mean()
	ax3.axhline(avg_table1c, color=G.CESA_BLUE, linestyle='--', label=f'Náš průměr: {avg_table1c:.2f}')
	ax3.axhline(avg_table2c, color=G.BUT_RED, linestyle='--', label=f'Elgendiho průměr: {avg_table2c:.2f}')
	#####################################################################
	######################################## SET AXIS ########################################
	ax3.set_ylabel('Rozdíl [tep/min]')
	# Show only every 8th label on the x-axis
	ax3.set_xticks(range(0, len(table1['df']['ID'][second_tr:]), 4))
	ax3.set_xticklabels([label for i, label in enumerate(table1['df']['ID'][second_tr:]) if i % 4 == 0])
	# Add small marks for the rest of the signals
	ax3.xaxis.set_minor_locator(plt.MultipleLocator(1))
	ax3.tick_params(axis='x', which='minor', length=4, color='gray')
	###########################################################################################
	ax3.margins(x=0, y=0.1)
	ax3.legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize='small')
	ax3.tick_params(axis='x', rotation=90)

	plt.tight_layout()
	plt.show()


def full_results(print_head=True):
	"""
	Reads a file containing multiple CSV-style tables with repeated headers,
	splitting them into separate DataFrames.
	
	Each table is preceded by a line that might act as a 'title' (e.g. "CB My:"),
	or starts immediately with a header line.
		
	Returns
	-------
	tables : list of dict
		A list of dictionaries, each with keys:
		  - "title": str or None, a label for the table (if we detect it)
		  - "df": pd.DataFrame containing that sub-table's data
	"""
	with open('./results.csv', 'r') as f:
		lines = f.read().splitlines()

	tables, current_table_lines = [], []
	current_title = None

	# Helper function to create a DataFrame from the lines collected so far
	def flush_current_table(title, table_lines):
		"""Parses the accumulated lines into a DataFrame, returns dict with title & df."""
		# If there is no valid CSV content, just skip
		if not table_lines:
			return None
		# Try reading it with pandas
		try:
			df = pd.read_csv(StringIO("\n".join(table_lines)))
			return {"title": title, "df": df}
		except Exception as e:
			# If something goes wrong (e.g., lines aren't valid CSV), skip or log
			print(f"Skipping invalid block under title '{title}': {e}")
			return None

	for line in lines:
		line_stripped = line.strip()
		
		# 1: If a line ends with a colon (e.g. "CB My:"), treat it as a new "section title".
		# Adjust the condition to match your real CSV structure.
		if line_stripped.endswith(":"):
			# If there are lines accumulated for the previous table, flush them
			table_data = flush_current_table(current_title, current_table_lines)
			if table_data:
				tables.append(table_data)
			# Reset and treat this line as title
			current_title = line_stripped[:-1].strip()  # remove ':' and any surrounding whitespace
			current_table_lines = []
		
		# 2: If we detect a new CSV header line (e.g. line starts with "ID," etc.)
		# we assume a new sub-table is starting. The previous lines, if any, are a separate table.
		elif (line_stripped.startswith("ID,")):
			# If there’s an existing block, flush it as a table
			table_data = flush_current_table(current_title, current_table_lines)
			if table_data:
				tables.append(table_data)
			# Start collecting lines for the new table
			current_table_lines = [line]
		else:
			# Otherwise, keep accumulating lines in the current block
			current_table_lines.append(line)

	# End of file—flush any remaining lines
	table_data = flush_current_table(current_title, current_table_lines)
	if table_data:
		tables.append(table_data)

	# Inspect each sub-table = we can easily check and fix any problems
	# if print_head:
	# 	for table in tables:
	# 		print(f" Title: {table['title']}")
	# 		print(table['df'].head())
	# 		print("\n====================================================================================")

	return tables