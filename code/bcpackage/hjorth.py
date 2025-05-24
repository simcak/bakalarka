from bcpackage import globals as G
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

###############################################################
##################### SHOWING && PLOTTING #####################
###############################################################

def confusion_matrix(database='all'):
	import seaborn as sns
	from sklearn.metrics import confusion_matrix

	if database == 'CapnoBase30':
		data = pd.read_csv('./hjorth_CBq.csv')
	elif database == 'CapnoBase300':
		data = pd.read_csv('./hjorth_CBq_300.csv')
	elif database == 'BUT_PPG':
		data = pd.read_csv('./hjorth_butppg.csv')
	elif database == 'all':
		data_cb = pd.read_csv('./hjorth_CBq.csv')
		data_but = pd.read_csv('./hjorth_butppg.csv')
		data = pd.concat([data_cb, data_but], ignore_index=True)
	else:
		raise ValueError("Invalid database name. Choose 'CapnoBase30', 'CapnoBase300', 'BUT_PPG', or 'all'.")

	# Calculate confusion matrix
	cm = confusion_matrix(
		data['ourQ_this_only'],
		(data['Orphanidou Quality'] >= 0.9).astype(int)
	)

	# Plot confusion matrix with colors
	plt.figure(figsize=(6, 5))
	sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
				xticklabels=["Bad Quality", "Good Quality"], 
				yticklabels=["Bad Quality", "Good Quality"])
	plt.xlabel("Predicted Quality")
	plt.ylabel("True Quality")
	plt.title("Confusion Matrix")
	plt.tight_layout()
	plt.show()

def hjorth_show_hr(chunked_pieces, database='CapnoBase300'):
	# Load the data from the CSV file
	if database == 'CapnoBase30':
		data = pd.read_csv('./hjorth_CBq.csv')
	elif database == 'CapnoBase300':
		data = pd.read_csv('./hjorth_CBq_300.csv')
	elif database == 'BUT_PPG':
		data = pd.read_csv('./hjorth_butppg.csv')
	else:
		raise ValueError("Invalid database name. Choose 'CapnoBase30', 'CapnoBase300', 'BUT_PPG'.")

	# Extract relevant columns
	file_names = data['File name']
	hjorth_hr = data['Hjorth HR']
	ref_hr = data['Ref HR']
	hr_diff = data['HR diff']

	average_hr_diff = data['HR diff'].mean()
	print(f"\033[92m\033[1mPrůměrný rozdíl TF (tepů/min): {average_hr_diff:.2f}\033[0m")

	# Select every x-th file name for labeling (x = chunked_pieces)
	selected_indices = range(0, len(file_names), chunked_pieces)

	####################### Plot the results #######################
	################################################################
	plt.figure(figsize=(12, 6))

	# Plot Hjorth HR and Ref HR
	plt.subplot(2, 1, 1)
	plt.title('Hjorthova TF vs Referenční TF')
	plt.plot(hjorth_hr, label='Hjorthova TF (tepů/min)', marker='o')
	plt.plot(ref_hr, label='Referenční TF (tepů/min)', marker='x')
	plt.xlabel('Index signálu')
	plt.ylabel('Srdeční frekvence (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hjorth_hr.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	# Plot HR difference
	plt.subplot(2, 1, 2)
	plt.title('Rozdíl TF (Hjorthova TF - Referenční TF)')
	plt.plot(hr_diff, label='Rozdíl TF (tepů/min)', color='red', marker='s')
	plt.xlabel('Index signálu')
	plt.ylabel('|Δ TF| (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hr_diff.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	plt.tight_layout()
	plt.show()
	################################################################

def hjorth_show_only_quality_hr(chunked_pieces, database='CapnoBase300'):
	# data = data[data['Orphanidou Quality'] >= 0.9]
	# data = data[data['HR diff'] <= 100]

	# Load the data from the CSV file
	if database == 'CapnoBase30':
		data = pd.read_csv('./hjorth_CBq.csv')
		data = data[data['Our Quality'] == 1]
	elif database == 'CapnoBase300':
		data = pd.read_csv('./hjorth_CBq_300.csv')
		data = data[data['ourQ_this_only'] == 1]
	elif database == 'BUT_PPG':
		data = pd.read_csv('./hjorth_butppg.csv')
		data = data[data['Our Quality'] == 1]
	else:
		raise ValueError("Invalid database name. Choose 'CapnoBase30', 'CapnoBase300', 'BUT_PPG'.")

	# Extract relevant columns
	file_names = data['File name']
	hjorth_hr = data['Hjorth HR']
	ref_hr = data['Ref HR']
	hr_diff = data['HR diff']

	average_hr_diff = hr_diff.mean()
	print(f"\033[92m\033[1mPrůměrný rozdíl TF (tepů/min): {average_hr_diff:.2f}\033[0m")

	# Select every x-th file name for labeling (x = chunked_pieces)
	selected_indices = range(0, len(file_names), chunked_pieces)

	####################### Plot the results #######################
	################################################################
	plt.figure(figsize=(12, 6))

	# Plot Hjorth HR and Ref HR
	plt.subplot(2, 1, 1)
	plt.title('Hjorthova TF vs Referenční TF (Our Quality = 1)')
	plt.plot(hjorth_hr.values, label='Hjorthova TF (tepů/min)', marker='o')
	plt.plot(ref_hr.values, label='Referenční TF (tepů/min)', marker='x')
	plt.xlabel('Index signálu')
	plt.ylabel('Srdeční frekvence (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hjorth_hr.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	# Plot HR difference
	plt.subplot(2, 1, 2)
	plt.title('Rozdíl TF (Hjorthova TF - Referenční TF) (Our Quality = 1)')
	plt.plot(hr_diff.values, label='Rozdíl TF (tepů/min)', color='red', marker='s')
	plt.xlabel('Index signálu')
	plt.ylabel('|Δ TF| (tepů/min)')
	plt.legend()
	plt.grid()

	# Add vertical labels for selected file names
	for idx in selected_indices:
		plt.text(idx, hr_diff.iloc[idx], file_names.iloc[idx], rotation=90, fontsize=8, ha='center')

	plt.tight_layout()
	plt.show()
	################################################################

def hjorth_show_spi(database='CapnoBase300'):
	# Plot the relationship between SPI and HR difference
	if database == 'CapnoBase30':
		data = pd.read_csv('./hjorth_CBq.csv')
	elif database == 'CapnoBase300':
		data = pd.read_csv('./hjorth_CBq_300.csv')
	elif database == 'BUT_PPG':
		data = pd.read_csv('./hjorth_butppg.csv')
	else:
		raise ValueError("Invalid database name. Choose 'CapnoBase30', 'CapnoBase300', 'BUT_PPG'.")

	plt.figure(figsize=(8, 6))
	plt.scatter(data["SPI Filtered"], data["HR diff"], alpha=0.7, c=G.CESA_BLUE, edgecolors='k')

	# Popisky a vzhled
	plt.title("Závislost rozdílu TF na SPI", fontsize=14)
	plt.xlabel("Spectral Purity Index (SPI)", fontsize=12)
	plt.ylabel("Rozdíl TF [bpm]", fontsize=12)
	plt.grid(True)
	plt.tight_layout()

	# Zobraz
	plt.show()

###############################################################
######################### ML QUALITY ##########################
###############################################################

def quality_hjorth(find_best_parameters=False, database='all'):
	from bcpackage import time_count
	import seaborn as sns
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.model_selection import train_test_split, cross_val_score
	from sklearn.preprocessing import StandardScaler
	from sklearn.metrics import classification_report, confusion_matrix

	def _plot_feature_importance(_X_df, _trained_model):
		# Library of feature thresholds with keynames
		feature_thresholds = {feature_name: [] for feature_name in _X_df.columns}

		# Go through all trees in the forest
		for tree_estimator in _trained_model.estimators_:
			tree = tree_estimator.tree_
			
			# Go through all nodes in the tree
			for i in range(tree.node_count):
				# Check if the node is a leaf
				if tree.feature[i] != -2:
					feature_idx = tree.feature[i]
					threshold_scaled = tree.threshold[i] # This is the threshold used in the tree
					# Add the threshold to the list for the corresponding feature
					feature_name = X_df.columns[feature_idx]
					feature_thresholds[feature_name].append(threshold_scaled)

		# Check threashold for each feature
		for feature_name, thresholds_list in feature_thresholds.items():
			if thresholds_list:
				print(f"\nPrahové hodnoty pro příznak: {feature_name} (na škálovaných datech)")
				print(f"  Počet použití:  {len(thresholds_list)}")
				print(f"  Minimální práh: {np.min(thresholds_list):.4f}")
				print(f"  Mediánový práh: {np.median(thresholds_list):.4f}")
				print(f"  Průměrný práh:  {np.mean(thresholds_list):.4f}")
				print(f"  Maximální práh: {np.max(thresholds_list):.4f}")

				plt.figure(figsize=(10, 4))
				sns.histplot(thresholds_list, kde=True, bins=20)
				plt.title(f"Distribuce prahů pro {feature_name} (škálovaná data)")
				plt.xlabel("Prahová hodnota (škálovaná)")
				plt.ylabel("Frekvence")
				plt.show()
			else:
				print(f"\nPro příznak {feature_name} nebyly v RF použity žádné přímé rozhodovací prahy (může být méně důležitý nebo použit v kombinaci).")

	def _find_best_parameters(_X, _y):
		"""
		Printing out the best parameters for Random Forest Classifier that we use below.
		"""
		from sklearn.model_selection import GridSearchCV

		param_grid = {
			"n_estimators": [10, 25, 50, 100, 200],
			"max_depth": [None, 2, 5, 10],
			"max_features": ["sqrt", "log2", None],
		}

		search = GridSearchCV(RandomForestClassifier(class_weight="balanced", random_state=42), param_grid, cv=5, scoring="f1")
		search.fit(_X, _y)
		print(f"Nejlepší model: {search.best_estimator_}")
		print(f"Nejlepší F1 skóre: {search.best_score_:.3f}")
		print(f"Nejlepší parametry: {search.best_params_}")
		
		# Return the best parameters
		return search.best_params_["max_depth"], search.best_params_["max_features"], search.best_params_["n_estimators"]

	# Start the timer
	start_time, stop_event = time_count.terminal_time()

	### Load the data from the CSV files
	if database == 'CapnoBase30':
		df = pd.read_csv("./hjorth_CBq.csv")
		df["source"] = "capno"
	elif database == 'CapnoBase300':
		df = pd.read_csv("./hjorth_CBq_300.csv")
		df["source"] = "capno"
	elif database == 'BUT_PPG':
		df = pd.read_csv("./hjorth_butppg.csv")
		df["source"] = "but"
	elif database == 'all':
		df_capno = pd.read_csv("./hjorth_CBq.csv")
		df_capno["source"] = "capno"
		df_butppg = pd.read_csv("./hjorth_butppg.csv")
		df_butppg["source"] = "but"
		df = pd.concat([df_capno, df_butppg], ignore_index=True)
	else:
		raise ValueError("Invalid database name. Choose 'CapnoBase30', 'CapnoBase300', 'BUT_PPG', or 'all'.")

	### Define what will we use for classification
	features = ["Mobility Filtered", "Complexity Filtered", "SPI Filtered"]
	# features = ["SPI Filtered"]
	X_df = df[features]

	### Define the target variable
	target = (df["Orphanidou Quality"] >= 0.9).astype(int)

	### Scaling
	scaler = StandardScaler()
	X_scaled = scaler.fit_transform(X_df)

	### Split the data into training and testing sets
	X_train, X_test, y_train, y_test = train_test_split(X_scaled, target, stratify=target, test_size=0.4, random_state=42)

	if find_best_parameters:
		_max_depth, _max_features, _n_estimators = _find_best_parameters(X_train, y_train)
	else:
		_max_depth, _max_features, _n_estimators = 5, None, 50

	### Model training
	trained_model = RandomForestClassifier(
		n_estimators=_n_estimators, max_features=_max_features, max_depth=_max_depth, random_state=42, class_weight="balanced"
		)
	trained_model.fit(X_train, y_train)

	### Cross-validation: evaluating estimator performance
	_fold = 5
	scores = cross_val_score(trained_model, X_scaled, target, cv=_fold, scoring="f1")
	print(f"Průměrné F1 skóre ({_fold}-fold CV): {scores.mean():.3f} ± {scores.std():.3f}")

	### predicting on the test set
	y_pred = trained_model.predict(X_test)

	### Print && show
	# _plot_feature_importance(X_df, trained_model)
	print(f"Klasifikační zpráva ({database}):")
	print(classification_report(y_test, y_pred, digits=3))
	print(f"Matice záměn ({database}):")
	print("TN   FP\nFN   TP")
	print(confusion_matrix(y_test, y_pred))

	# Vyhodnocení zvlášť pro CapnoBase a BUT_PPG podle 'source'
	if database == 'all':
		for src in ["capno", "but"]:
			mask = X_test[:, 0] == X_test[:, 0]  # dummy mask, will be replaced
			if "source" in df.columns:
				test_indices = X_test.shape[0]
				# Najdi indexy v původním dataframe, které odpovídají X_test
				# Protože jsme použili train_test_split na X_scaled, musíme najít odpovídající řádky v df
				# Uděláme to přes indexy
				_, X_test_idx = train_test_split(
					df.index, stratify=target, test_size=0.4, random_state=42
				)
				source_mask = df.iloc[X_test_idx]["source"] == src
				y_test_src = y_test[source_mask.values]
				y_pred_src = y_pred[source_mask.values]
				print(f"\nKlasifikační zpráva ({src}):")
				print(classification_report(y_test_src, y_pred_src, digits=3))
				print(f"Matice záměn ({src}):")
				print(confusion_matrix(y_test_src, y_pred_src))

	# Stop the timer and print the elapsed time
	time_count.stop_terminal_time(start_time, stop_event, func_name='Random Forest Classifier')

	return trained_model.predict(X_scaled)

# def quality_hjorth():
# 	from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
# 	from sklearn.model_selection import train_test_split
# 	from sklearn.metrics import classification_report, confusion_matrix
# 	from sklearn.preprocessing import StandardScaler

# 	# Define sources
# 	df_capno = pd.read_csv("./hjorth_CBq.csv")
# 	df_capno["source"] = "capno"

# 	df_butppg = pd.read_csv("./hjorth_butppg.csv")
# 	df_butppg["source"] = "but"

# 	df = pd.concat([df_capno, df_butppg], ignore_index=True)

# 	# Define what will we use for classification
# 	features = ["Mobility Filtered",
# 				"Complexity Filtered",
# 				"SPI Filtered",
# 				"Spectral Ratio",
# 				"ACF Peaks",
# 				"Shannon Entropy"
# 				]
# 	X = df[features]
# 	y = (df["Orphanidou Quality"] >= 0.9).astype(int)

# 	# Scaling for uniformity of Hjorth parameters
# 	scaler = StandardScaler()
# 	X_scaled = scaler.fit_transform(X)

# 	# Split the data into training and testing sets
# 	X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, stratify=y, test_size=0.2, random_state=66)

# 	# Model training
# 	_max_depth = 4
# 	clf = RandomForestClassifier(n_estimators=100, random_state=6)
# 	# clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=_max_depth, random_state=6)
# 	clf.fit(X_train, y_train)

# 	y_pred = clf.predict(X_test)

# 	# Get feature importances
# 	importances = clf.feature_importances_

# 	# Sort the features by importance
# 	sorted_indices = np.argsort(importances)[::-1]

# 	print("Důležitost příznaků podle Random Forest:")
# 	for i in sorted_indices:
# 		print(f"{features[i]}: {importances[i]:.4f}")

# 	# Print && show)
# 	print("Klasifikační zpráva:")
# 	print(classification_report(y_test, y_pred, digits=3))
# 	print("Matice záměn:")
# 	print(confusion_matrix(y_test, y_pred))

# 	def _plot_confusion_matrix(y_true, y_pred, class_names=["Špatná kvalita", "Dobrá kvalita"]):
# 		import seaborn as sns
# 		from sklearn.metrics import classification_report, confusion_matrix

# 		# Confusion matrix
# 		cm = confusion_matrix(y_true, y_pred)
# 		plt.figure(figsize=(6, 5))
# 		sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
# 					xticklabels=class_names, yticklabels=class_names)
# 		plt.xlabel("Predikovaná třída")
# 		plt.ylabel("Skutečná třída")
# 		plt.title("Matice záměn (Confusion Matrix)")
# 		plt.tight_layout()
# 		plt.show()

# 		# Classification report
# 		print("\nKlasifikační zpráva:")
# 		print(classification_report(y_true, y_pred, target_names=class_names))

# 	def _plot_feature_importance(clf, feature_names):
# 		importances = clf.feature_importances_
# 		indices = np.argsort(importances)[::-1]

# 		plt.figure(figsize=(10, 6))
# 		plt.title("Důležitost příznaků (Random Forest)")
# 		plt.bar(range(len(importances)), importances[indices], align="center")
# 		plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
# 		plt.tight_layout()
# 		plt.grid(True, linestyle='--', alpha=0.5)
# 		plt.show()

# 	def _plot_2d_projection(X, y, source, method='pca', title='2D projekce dat'):
# 		from sklearn.decomposition import PCA
# 		from sklearn.manifold import TSNE
# 		import seaborn as sns

# 		# Dimenzionální redukce
# 		if method == 'pca':
# 			X_proj = PCA(n_components=2).fit_transform(X)
# 		elif method == 'tsne':
# 			X_proj = TSNE(n_components=2, perplexity=30, n_iter=1000).fit_transform(X)
# 		else:
# 			raise ValueError("Podporované metody: 'pca', 'tsne'")

# 		# Příprava DataFrame pro seaborn
# 		df_plot = pd.DataFrame({
# 			"X1": X_proj[:, 0],
# 			"X2": X_proj[:, 1],
# 			"Třída": y,
# 			"Databáze": source
# 		})

# 		plt.figure(figsize=(8, 6))
# 		sns.scatterplot(
# 			data=df_plot,
# 			x="X1", y="X2",
# 			hue="Třída",       # barva = kvalita
# 			style="Databáze",  # tvar = zdroj dat
# 			palette=["#d62728", "#1f77b4"],
# 			alpha=0.7
# 		)
# 		plt.title(title)
# 		plt.xlabel("1. komponenta")
# 		plt.ylabel("2. komponenta")
# 		plt.legend(title="Třída / Databáze", loc="best", fontsize=9)
# 		plt.grid(True, linestyle='--', alpha=0.5)
# 		plt.tight_layout()
# 		plt.show()

# 	# _plot_confusion_matrix(y_test, y_pred)
# 	# _plot_feature_importance(clf, features)
# 	# _plot_2d_projection(X_scaled, y, df["source"], method='pca', title='2D projekce dat (PCA)')
# 	# _plot_2d_projection(X_scaled, y, df["source"], method='tsne', title='2D projekce dat (t-SNE)')

###############################################################
#################### SUPPORTING FUNCTIONS #####################
###############################################################

def _normalize_signal(signal):
	"""
	Normalize the signal to range [-1, 1].
	"""
	normalized_signal = signal / np.max(np.abs(signal)) if np.max(np.abs(signal)) > 0 else signal
	return normalized_signal

def _standardize_signal(signal):
	"""
	Standardize the signal to have mean 0 and standard deviation 1.
	"""
	standardized_signal = (signal - np.mean(signal)) / np.std(signal) if np.std(signal) > 0 else signal
	return standardized_signal

def _autocorrelate_signal(signal, num_iterations=1):
	from scipy.signal import correlate
	"""
	Perform repeated autocorrelation on the input signal.
	Autocorrelation is a method to find repeating patterns in the signal.

	Autocorrelation VS Correlation:
	- Autocorrelation: Correlation of a signal with itself at different time lags.
	- Correlation: Correlation of two different signals.
	"""
	result = signal.copy()
	for _ in range(num_iterations):
		result = correlate(result, result, mode='same', method='fft')	# O(N log N)
		# result = np.correlate(result, result, mode='same')			# O(i * N^2)
	return result

def _highpass_filter(signal, cutoff_frequency, sampling_frequency, order=4):
	"""
	Apply a high-pass Butterworth filter to the input signal.
	"""
	from scipy.signal import butter, filtfilt

	nyquist_frequency = 0.5 * sampling_frequency
	normalized_cutoff = cutoff_frequency / nyquist_frequency
	b, a = butter(order, normalized_cutoff, btype='high', analog=False)
	filtered_signal = filtfilt(b, a, signal)
	return filtered_signal

def _lowpass_filter(signal, cutoff_frequency, sampling_frequency, order=4):
	"""
	Apply a low-pass Butterworth filter to the input signal.
	"""
	from scipy.signal import butter, filtfilt

	nyquist_frequency = 0.5 * sampling_frequency
	normalized_cutoff = cutoff_frequency / nyquist_frequency
	b, a = butter(order, normalized_cutoff, btype='low', analog=False)
	filtered_signal = filtfilt(b, a, signal)

	return filtered_signal

###############################################################
####################### HJORTH ALGORITHM ######################
###############################################################

def compute_hjorth_parameters(data, index, autocorr_iterations, only_quality=False):
	"""
	Compute Hjorth parameters: mobility, complexity, and spectral purity index (SPI).

	Parameters:
	- signal: numpy array (n x 1), scalar time series
	- sampling_frequency: float, sampling frequency [Hz]

	Returns:
	- mobility_hz: dominant frequency (mobility / (2 * pi * T)) [Hz]
	- complexity_hz: bandwidth (complexity / (2 * pi * T)) [Hz]
	- spectral_purity_index: SPI, value between 0 and 1, 1 means pure harmonic
	"""
	if only_quality and data["Orphanidou Quality"] is not None:
		if data["Orphanidou Quality"] < 0.9:	# for CapnoBase
			return None
		if data["Ref Quality"] is not None:		# for BUT PPG
			if data["Orphanidou Quality"] < 0.9 or data["Ref Quality"] == 0:
				return None

	file_id, signal, sampling_frequency, ref_hr = data["File name"], data["Raw Signal"], data["fs"], data["Ref HR"]
	orphanidou_quality = data["Orphanidou Quality"]
	our_quality = data["Our Quality"]

	# Name of the file
	if index == 0:
		file_name = file_id
	else:
		file_name = f"{file_id}_{index}min"

	# Compute sampling interval from frequency
	sampling_interval = 1 / sampling_frequency

	# Standardize the signal = remove DC component and keep the signal shape but still prevent from overflow later
	signal = _standardize_signal(signal)
	# Apply the high-pass filter to remove respiratory frequencies
	respiratory_cutoff_frequency = 0.5  # Example cutoff frequency for respiration [Hz] = equivalent to 30 bpm
	filtered_signal = _highpass_filter(signal, respiratory_cutoff_frequency, sampling_frequency)

	# Autocorrelate the signal
	autocorrelated_signal = _autocorrelate_signal(filtered_signal, num_iterations=autocorr_iterations)

	############################## HR #############################
	###############################################################
	def _hjorth_hr(signal, sampling_interval):
		# 1st derivative (velocity)
		first_derivative_hr = np.diff(signal)
		# Mean square values
		mean_square_derivative_hr = np.mean(first_derivative_hr ** 2)
		mean_square_signal_hr = np.mean(signal ** 2)
		# Hjorth MOBILITY
		mobility_hr = np.sqrt(mean_square_derivative_hr / mean_square_signal_hr) if mean_square_signal_hr > 0 else 0
		# Convert to Hz (dominant frequency)
		mobility_hz_hr = mobility_hr / (2 * np.pi * sampling_interval)

		return mobility_hz_hr

	mobility_hz_autocorr = _hjorth_hr(autocorrelated_signal, sampling_interval)

	########################### Quality ###########################
	###############################################################
	def _hjorth_quality(signal, fs):
		import numpy as np

		# Lowpass filtr (pro SPI)
		signal = _lowpass_filter(signal, 3.35, fs)

		# Derivace
		first_derivative_q = np.diff(signal)
		second_derivative_q = np.diff(first_derivative_q)

		# Výpočty Hjorth deskriptorů
		mean_square_signal_q = np.mean(signal ** 2)
		mean_square_derivative_q = np.mean(first_derivative_q ** 2)
		mean_square_second_derivative_q = np.mean(second_derivative_q ** 2)

		div_1 = mean_square_derivative_q / mean_square_signal_q if mean_square_signal_q > 0 else 0
		div_2 = mean_square_second_derivative_q / mean_square_derivative_q if mean_square_derivative_q > 0 else 0

		mobility_q = np.sqrt(div_1) if div_1 > 0 else 0
		complexity_q = np.sqrt(div_2 / div_1) if div_1 > 0 and div_2 > 0 else 0
		spi_q = np.sqrt(div_1 / div_2) if div_1 > 0 and div_2 > 0 else 0

		return mobility_q, complexity_q, spi_q

	if orphanidou_quality is not None:
		mobility, complexity, spi = _hjorth_quality(filtered_signal, sampling_frequency)
	else:
		mobility, complexity, spi = None, None, None
	###############################################################

	hjorth_info = {
		"File name": file_name,
		"Domain Freq [Hz]": mobility_hz_autocorr,
		"Hjorth HR": mobility_hz_autocorr * 60,
		"Ref HR": ref_hr,
		"HR diff": abs(ref_hr - (mobility_hz_autocorr * 60)),
		"Mobility Filtered": mobility,
		"Complexity Filtered": complexity,
		"SPI Filtered": spi,
		"Orphanidou Quality": orphanidou_quality,
		"Our Quality": our_quality,
		"Ref Quality": data["Ref Quality"],
	}

	_export = True
	if _export:
		output_file = "./hjorth.csv"
		try:
			hjorth_df = pd.read_csv(output_file)
		except FileNotFoundError:
			hjorth_df = pd.DataFrame()

		hjorth_df = pd.concat([hjorth_df, pd.DataFrame([hjorth_info])], ignore_index=True)
		hjorth_df.to_csv(output_file, header=True, index=False)

	_show = False
	if _show and hjorth_info["HR diff"] > 5:
		# Frekvenční charakteristika (FFT)
		freqs = np.fft.rfftfreq(len(filtered_signal), d=1/sampling_frequency)
		fft_magnitude = np.abs(np.fft.rfft(filtered_signal))
		freqs_autocorr = np.fft.rfftfreq(len(autocorrelated_signal), d=1/sampling_frequency)
		fft_magnitude_autocorr = np.abs(np.fft.rfft(autocorrelated_signal))
		freqs_raw = np.fft.rfftfreq(len(signal), d=1/sampling_frequency)
		fft_magnitude_raw = np.abs(np.fft.rfft(signal))

		# Rescale FFT magnitudes to the same scale
		fft_magnitude_rescaled = fft_magnitude / np.max(fft_magnitude)
		fft_magnitude_autocorr_rescaled = fft_magnitude_autocorr / np.max(fft_magnitude_autocorr)
		fft_magnitude_raw_rescaled = fft_magnitude_raw / np.max(fft_magnitude_raw)

		# Vykreslení autokorelovaného signálu a derivace
		plt.figure(figsize=(12, 6))

		plt.subplot(2, 1, 1)
		plt.title("Přeškálovaný filtrovaný a autokorelovaný signál")
		plt.xlabel("Čas [s]")
		plt.ylabel("Relativní amplituda") # je amplituda správně použité slovo?
		time_axis = np.arange(len(filtered_signal)) / sampling_frequency
		plt.plot(time_axis, _normalize_signal(filtered_signal), label="Filtrovaný signál")
		plt.plot(time_axis, _normalize_signal(autocorrelated_signal), label="Autokorelovaný signál")
		plt.legend()
		plt.grid()

		# Vykreslení frekvenční charakteristiky
		plt.subplot(2, 1, 2)
		freq_max = 15
		plt.plot(freqs[freqs <= freq_max], fft_magnitude_rescaled[freqs <= freq_max], label="Frekvenční charakteristika filtrovaného signálu")
		plt.plot(freqs_autocorr[freqs_autocorr <= freq_max], fft_magnitude_autocorr_rescaled[freqs_autocorr <= freq_max], label="Frekvenční charakteristika autokorelovaného signálu")
		plt.plot(freqs_raw[freqs_raw <= freq_max], fft_magnitude_raw_rescaled[freqs_raw <= freq_max], label="Frekvenční charakteristika původního signálu", color='grey', alpha=0.5)
		plt.axvline(x=hjorth_info["Domain Freq [Hz]"], color=G.CESA_BLUE, linestyle='--', label=f"Mobilita: {hjorth_info['Domain Freq [Hz]']:.2f} Hz")
		plt.title("Přeškálovaná frekvenční charakteristika signálů")
		plt.ylabel("Relativní amplituda")
		plt.xlabel("Frekvence [Hz]")
		plt.legend()
		plt.grid()

		plt.tight_layout()
		plt.show()

	return hjorth_info

def hjorth_alg(database, chunked_pieces=1, autocorr_iterations=5, compute_quality=False):
	"""
	Calculate by Hjorth parameters the HR for the given database and evaluate the results.
	"""
	from bcpackage import time_count
	from bcpackage.capnopackage import cb_data
	from bcpackage.butpackage import but_data, but_error
	import neurokit2 as nk

	def _downsample_signal(file_info, target_fs):
		"""
		Downsample the signal to the target sampling frequency.
		"""
		from scipy.signal import resample

		if file_info['fs'] == target_fs:
			return file_info

		num_samples = int(len(file_info['Raw Signal']) * target_fs / file_info['fs'])
		resampled_signal = resample(file_info['Raw Signal'], num_samples)
		resampled_ref_peaks = np.array(file_info['Ref Peaks']) * target_fs / file_info['fs']

		file_info['Raw Signal'] = resampled_signal
		file_info['fs'] = target_fs
		file_info['Ref Peaks'] = resampled_ref_peaks

		return file_info

	def _chunking_signal(chunked_pieces, file_info, chunk_idx):
		"""
		Chunk the signal into smaller segments for processing them.
		Last chunk may be longer than the others.
		Chunking is only supported for CapnoBase database.
		"""
		if chunked_pieces == 1:
			return file_info['Raw Signal'], file_info['Ref HR']

		import numpy as np
		from bcpackage import calcul
		# Calculate the length of each chunk in samples
		chunk_len = len(file_info['Raw Signal']) // chunked_pieces

		start_idx = chunk_idx * chunk_len
		end_idx = start_idx + chunk_len		# ignoring the last chunk if it is not full
		# if chunk_idx == chunked_pieces - 1:
		# 	end_idx = len(file_info['Raw Signal'])
		# else:
		# 	end_idx = start_idx + chunk_len

		ppg_chunk = file_info['Raw Signal'][start_idx:end_idx]

		chunk_ref_peaks = np.array(file_info['Ref Peaks'])
		mask = (chunk_ref_peaks >= start_idx) & (chunk_ref_peaks < end_idx)
		chunk_ref_peaks = chunk_ref_peaks[mask] - start_idx

		hr_info = calcul.heart_rate(chunk_ref_peaks, None, file_info['fs'], init=True)
		chunk_ref_hr = hr_info['Ref HR']

		return ppg_chunk, chunk_ref_hr

	def _prepare_data_for_hjorth(_name, _fs, _raw_signal, _ref_quality, _ref_hr, calculate_orphanidou_quality=False):
		"""
		Prepare the data for Hjorth parameters calculation.
		"""
		if calculate_orphanidou_quality:
			nk_signals, info = nk.ppg_process(_raw_signal, sampling_rate=_fs, method_quality="templatematch") # Orphanidou method
			_orphanidou_quality = np.mean(nk_signals['PPG_Quality'])
		else:
			_orphanidou_quality = None

		data = {
			"File name": _name,
			"fs": _fs,
			"Raw Signal": _raw_signal,
			"Orphanidou Quality": _orphanidou_quality,
			"Ref Quality": _ref_quality,
			"Ref HR": _ref_hr,
		}
		return data

	our_quality = quality_hjorth()
	# Start the timer
	start_time, stop_event = time_count.terminal_time()

	if database == "CapnoBase":
		for i in range(G.CB_FILES_LEN):
			file_info_original = cb_data.extract(G.CB_FILES[i])
			file_info = _downsample_signal(file_info_original, 30) 	# downsample to 30Hz = same as BUT PPG
			max_chunk_count = len(file_info['Raw Signal']) // (file_info['fs'] * 10) # 10s long chunks
			# Chunk the signal
			if chunked_pieces >= 1 and chunked_pieces <= max_chunk_count:
				for j in range(chunked_pieces):
					# Preparing data
					chunked_singal, chunk_ref_hr = _chunking_signal(chunked_pieces, file_info, j)
					_data = _prepare_data_for_hjorth(
						file_info["ID"], file_info["fs"], chunked_singal, None, chunk_ref_hr,
						calculate_orphanidou_quality=compute_quality
					)
					_data["Our Quality"] = our_quality[i * chunked_pieces + j]
					compute_hjorth_parameters(_data, j, autocorr_iterations, only_quality=False)
			else:
				raise ValueError(f"\033[91m\033[1mInvalid chunk value. Use values in range <1 ; {max_chunk_count}> == <hole signal ; 10s long chunks>\033[0m")

	elif database == "BUT_PPG":
		j = 0
		for i in range(G.BUT_DATA_LEN):
			if chunked_pieces == 1:
				file_info = but_data.extract(i)
				if but_error.police(file_info, i):
					continue
				# Preparing data
				_data = _prepare_data_for_hjorth(
					file_info['ID'], file_info['PPG_fs'], file_info['PPG_Signal'], file_info['Ref_Quality'], file_info['Ref_HR'],
					calculate_orphanidou_quality=compute_quality
				)
				_data["Our Quality"] = our_quality[j + 2016]
				j += 1
				compute_hjorth_parameters(_data, 0, autocorr_iterations, only_quality=False)
			else:
				raise ValueError("\033[91m\033[1mChunking is not supported for BUT PPG database.\033[0m")
		chunked_pieces = 10

	else:
		raise ValueError("\033[91m\033[1mInvalid database. Use 'CapnoBase' or 'BUT_PPG'.\033[0m")

	# Stop the timer and print the elapsed time
	time_count.stop_terminal_time(start_time, stop_event, func_name=f'Hjorth - {database}')
