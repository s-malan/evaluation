# ASR Evaluation

## Overview

This repository evaluates ASR metrics falling in the word segmentation and clustering categories. The metrics covered are: Precision, Recall, F1-Score, Over-Segmentation, R-Value, normalized edit distance (NED), and coverage. More information on these metrics can be found in ASR_metrics.md. Both scripts can be called as main methods, where segmentation and alignment files must be included, altermatively the word boundary scipt's functions can be called from within another scripts by supplying the functions with the correct variables (as noted in the function headers).

## Scripts

Note: all segmentation, alignment, and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Word Boundary Evaluation

Python script name: boundary_word.py

This script evaluates ASR word segmentation metrics namely: Precision, Recall, F1-Score, Over-Segmentation, and R-Value.

**Example Usage (as main)**

    python3 boundary_word.py path/to/segment/files path/to/alignment/files --alignment_format --frames_per_ms
where, alignment_format specifies the extension of the alignment files (options: .TextGrid, or .txt) and where frames_per_ms specifies the number of milliseconds contained in one frame for the audio encoding method used to find the segmentation. TODO: add tolerance and strict to input parameters.

### Cluster Evaluation

Python script name: ned_cov.py

This script evaluates ASR clustering metrics namely: coverage and NED.

**Example Usage (as main)**

    python3 ned_cov.py path/to/segment/files path/to/alignment/files --alignment_format
where, alignment_format is as specified above.