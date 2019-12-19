import timeit
import warnings

import matplotlib.pyplot as plt
import numpy as np

from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_rcv1

# We use SAGA solver
solver = 'saga'
model = 'multinomial'
max_iter = 1000
# Memorized fetch_rcv1 for faster access
data = fetch_rcv1()
X, y = data.data, data.target

n_samples = 10_000

if n_samples is not None:
    X, y = X[:n_samples], y[:n_samples]

# y is one-hot encoded and sparse for some reason ... needs to turn it into a vector of ints
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(data.target_names.reshape(-1, 1))
y = one_hot_encoder.inverse_transform(y)

ordinal_encoder = OrdinalEncoder()
y = ordinal_encoder.fit_transform(y)
# needs to be a 1d array and not a column vector (to avoid a warning)
y = np.ravel(y)

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    random_state=42,
                                                    # stratify=y.toarray(),
                                                    test_size=0.1)
train_samples, n_features = X_train.shape
n_classes = np.unique(y).shape[0]

print('Dataset 20newsgroup, train_samples=%i, n_features=%i, n_classes=%i'
      % (train_samples, n_features, n_classes))

lr = LogisticRegression(solver=solver,
                        multi_class=model,
                        penalty='l1',
                        # max_iter=
                        random_state=42,
                        )
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
accuracy = np.sum(y_pred == y_test) / y_test.shape[0]
density = np.mean(lr.coef_ != 0, axis=1) * 100

