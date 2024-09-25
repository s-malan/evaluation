# ASR Evaluation

This repository evaluates ASR metrics falling in the word segmentation and clustering categories. The metrics covered are: Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, R-Value, normalized edit distance (NED), coverage, and type scores. 
More information on these metrics can be found in ASR_metrics.md. Both scripts can be called as main methods, where segmentation and alignment files must be included, alternatively the word boundary script's functions can be called from within another scripts by supplying the functions with the correct variables (as noted in the function headers).

## Scripts

Note: all segmentation, alignment, and VAD files are saved in seconds and all file names containing time stamps are denoted in milliseconds.

### Word Boundary Evaluation

Python script name: boundary_word.py

This script evaluates ASR word segmentation metrics namely: Boundary Precision, Boundary Recall, Boundary F1-Score, Token Precision, Token Recall, Token F1-Score, Over-Segmentation, and R-Value.

**Example Usage**

    python3 boundary_word.py path/to/segment/files path/to/alignment/files --alignment_format=.TextGrid --frames_per_ms=20 --tolerance=1 --strict=True

The **alignment_format** argument specifies the extension of the alignment files (options: .TextGrid, or .txt).
The **frames_per_ms** argument specifies the number of milliseconds contained in one frame for the audio encoding method used to find the segmentation.
The **tolerance** argument specifies the number of frames (to both sides) that the hypothesized boundary can be from the ground truth boundary to still count.
The **strict** argument determines if the word boundary hit count is strict or lenient as described by D. Harwath in the following paper [https://ieeexplore.ieee.org/abstract/document/10022827](https://ieeexplore.ieee.org/abstract/document/10022827), default True.

The input file format is a .list file containing boundaries (and optional space-separated class assignment values) with each new value (or pair of values) on a new line.

### Cluster Evaluation

Python script name: ned_cov.py

This script evaluates ASR clustering metrics namely: NED, coverage, and type scores.

**Example Usage**

    python3 ned_cov.py path/to/segment/files path/to/alignment/files --alignment_format=.TextGrid

with the same argument definitions as above.

The input file format is a .list file containing boundaries and a  space-separated class assignment value with each new pair of values on a new line.

## Contributors

- Simon Malan
- [Benjamin van Niekerk](https://scholar.google.com/citations?user=zCokvy8AAAAJ&hl=en&oi=ao)
- [Herman Kamper](https://www.kamperh.com/)