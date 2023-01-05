import numpy as np

data = np.load("mnist.npz")
X = data['image'].astype(np.float32) / 255
y = np.eye(10)[data['label']]
X = X.reshape(X.shape[0], -1)

split = 50000

np.random.seed(0)

indices = np.random.permutation(X.shape[0])
training_idx, validation_idx = indices[:split], indices[split:]

X_train, X_validation = X[training_idx, :], X[validation_idx, :]
y_train, y_validation = y[training_idx, :], y[validation_idx, :]

np.savetxt("X_train.csv", X_train, delimiter = ',')
np.savetxt("X_test.csv", X_validation, delimiter = ',')
np.savetxt("y_train.csv", y_train, delimiter = ',')
np.savetxt("y_test.csv", y_validation, delimiter = ',')