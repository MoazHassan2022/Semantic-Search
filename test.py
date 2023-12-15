import numpy as np

rng = np.random.default_rng(50)
vectors = rng.random((1, 7), dtype=np.float32)

print(vectors)