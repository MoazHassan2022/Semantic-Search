import numpy as np

inverted_index = np.empty((14, 1024), dtype=set)

for i in range(14):
    for j in range(1024):
        inverted_index[i, j] = set()
        
inverted_index[0, 0].add(1)

print(inverted_index[2, 514])