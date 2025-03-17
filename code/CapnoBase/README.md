# Capnobase Recordings

This repository contains a set of recordings from the Capnobase database. Due to the large size of the data, the recordings are not uploaded directly to this repository.

## About Capnobase

Capnobase is a comprehensive database of capnography waveforms and related clinical data. It is used for research and development in the field of respiratory monitoring and analysis.

## Data Conversion

The recordings from the Capnobase database will be converted from `.mat` format to `.csv` format for easier accessibility and analysis.

## Data Structure

Read the data from either MATLAB or Octave.
The data are stored in the following structure (using first recording as an example):

```
0009_8min.mat/
├── SFresults/
│   ├── Fusion/
│   │   └── y [1x150 double]
│   ├── RIAV/
│   │   └── y [1x150 double]
│   ├── RIFV/
│   │   └── y [1x150 double]
│   ├── SmartFusion/
│   │   └── y [1x150 double]
│   └── x [1x150 double]
├── __refs_/
│   └── x [1x2 uint64]
├── labels/
│   ├── co2/
│   │   ├── artif/
│   │   │   └── x [1x2 uint64]
│   │   ├── startexp/
│   │   │   └── x [1x152 double]
│   │   └── startinsp/
│   │       └── x [1x152 double]
│   ├── ecg/
│   │   ├── artif/
│   │   │   └── x [1x2 uint64]
│   │   └── peak/
│   │       └── x [1x815 double]
│   ├── pleth/
│   │   ├── artif/
│   │   │   └── x [1x2 uint64]
│   │   └── peak/
│   │       └── x [1x816 double]
│   └── units/
│       └── x [1x7 uint16]
├── meta/
│   ├── persistentID [1x34 uint16]
│   ├── subject/
│   │   ├── age [1x1 double]
│   │   ├── gender [1x2 uint64]
│   │   └── weight [1x1 double]
│   └── treatment/
│       └── ventilation [1x11 uint16]
├── param/
│   ├── case/
│   │   ├── id [1x9 uint16]
│   │   └── ventilation [1x11 uint16]
│   └── samplingrate/
│       ├── co2 [1x1 double]
│       ├── ecg [1x1 double]
│       └── pleth [1x1 double]
├── reference/
│   ├── hr/
│   │   ├── ecg/
│   │   │   ├── x [1x814 double]
│   │   │   └── y [1x814 double]
│   │   └── pleth/
│   │       ├── x [1x814 double]
│   │       └── y [1x814 double]
│   ├── rr/
│   │   └── co2/
│   │       ├── x [1x151 double]
│   │       └── y [1x151 double]
│   └── units/
│       ├── hr/
│       │   └── y [1x9 uint16]
│       ├── rr/
│       │   └── y [1x11 uint16]
│       └── x [1x1 double]
└── signal/
    ├── co2/
    │   └── y [144001x1 double]
    ├── ecg/
    │   └── y [144001x1 double]
    ├── id [1x9 uint16]
    └── pleth/
        └── y [144001x1 double]

'SFresults': RR obtained using Smart Fusion approach, and steps thereof
'labels': labels for beats, breaths and artifacts obtained from a human rater
'meta': meta information such as demographics
'param': samplingrates and case name, ventilation mode
'reference': trends derived from labels
'signal': raw co2 and pleth signals
```