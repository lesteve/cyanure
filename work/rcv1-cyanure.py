import numpy as np

from sklearn.datasets import fetch_rcv1
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

import cyanure as cyan

data = fetch_rcv1()

X, y = data.data, data.target
yarr = y.toarray()

# y has two problems: it is one-hot encoded and is sparse. cyanure wants a
# dense vector of ints
one_hot_encoder = OneHotEncoder()
one_hot_encoder.fit(data.target_names.reshape(-1, 1))
y = one_hot_encoder.inverse_transform(y)

ordinal_encoder = OrdinalEncoder()
y = ordinal_encoder.fit_transform(y)

# normalize the rows of X in-place, without performing any copy
# cyan.preprocess(X, normalize=True, columns=False)
# declare a binary classifier for l2-logistic regression
classifier = cyan.MultiClassifier(loss='sqhinge', penalty='l2')
# classifier=cyan.BinaryClassifier(loss='sqhinge',penalty='l2')
# uses the auto solver by default, performs at most 500 epochs
# classifier.fit(X,y,lambd=0.1/X.shape[0],max_epochs=500,tol=1e-3,it0=5)
classifier.fit(X, y, lambd=0.1/X.shape[0], max_epochs=500, tol=1e-3, it0=5)
