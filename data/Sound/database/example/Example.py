import numpy as np
from get_data import read_data_full
X, y = read_data_full()
X = np.array(X)
y = np.array(y)

X = X.astype('float32') / 255.0
X = np.expand_dims(X, axis=-1)  # форма (N, 28, 28, 1)

print(X.shape)
print(y)