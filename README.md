# Implementation of NetSimile anomaly detection algorithm.
See https://cs.ucsb.edu/~victor/pub/ucsb/mae/references/berlingerio-netsimile-2012.pdf

## Required software:
Python 2.7
imported modules: networkx, statistics, scipy, matplotlib, os, sys

## Environmental variables:
None

## Instructions for running:
from command line: 
```python
python anomaly.py [relative path to input directory] [data outputfile] [image outputfile]
```
Example: 
```python
python anomaly.py anomaly/datasets/datasets/enron_by_day/ testout.txt test.png
```

NOTE: The edgelist reader automatically skips the first line of the file since many input
graphs contain node and edge sums as the first line.

## Results Interpretation:
- The distances between each adjacent time series vector are printed to the specified output file.
- Any anomalies (distances which exceed the given threshold) are printed to console.
- The time series plot of distances (and threshold) is written to the specified output file.
