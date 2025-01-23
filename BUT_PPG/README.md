# BUT PPG Recordings

This repository contains a set of recordings from the BUT PPG database. To avoid duplications, the original .mat recording inn't uploaded to this repository. Just the converted .csv file.

## About BUT PPG

The BUT PPG database is a collection of photoplethysmography (PPG) recordings from the Brno University of Technology. The recordings are used for research and development in the field of physiological monitoring and analysis.
The dataset contains 48 recordings of PPG and ECG signals, each with a duration of 100 seconds. It also includes the corresponding heart rate (HR) values, as well as the quality of the recordings.

## Data Conversion
From the `BUT_PPG.mat` file, the following data will be extracted/converted to the `.csv` file: `PPG`, `PPG_fs`, `ID`, `Quality`, `HR`.

## Data Structure

Read the data from either MATLAB or Octave.
The data are stored in the following structure:

```
BUT_PPG.mat/
├── ECG [48x10000 double]
├── PPG [48x300 double]
├── ECG_fs [1000]
├── PPG_fs [30]
├── ID [48x1 double]
├── Quality [48x1 double]
└── HR [48x1 double]
```