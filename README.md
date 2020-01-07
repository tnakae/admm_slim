# ADMM SLIM
This is experimental implementation of ADMM SLIM based on:

[ADMM SLIM: Sparse Recommendations for Many Users](http://www.cs.columbia.edu/~jebara/papers/wsdm20_ADMM.pdf)
in [WSDM 2020](http://www.wsdm-conference.org/2020/).

You can use two types of SLIM model:
- ADMM SLIM
- Dense SLIM

# Usage
1. Clone this repository in the directory where your script is placed.
``` bash
$ git clone https://github.com/tnakae/admm_slim/
```
2. Import package in your script and use it.
``` python
import numpy as np
from sklearn.model_selection import train_test_split

from admm_slim import AdmmSlim, DenseSlim

# Generate Sample Data
# Row : User, Column : Item
shape = [100, 40]
X = np.where(np.random.randn(*shape) > 1.0, 1, 0)

# Split data to train/test
X_train, X_test = train_test_split(X)

# Fit ADMM SLIM
# Change AdmmSlim to DenseSlim if you want to use Dense SLIM model
model = AdmmSlim()
model.fit(X_train)

# Predict
y_predict = model.predict(X_test)

# Top-n Item recommendation
y_top_10 = model.recommend(X_test, top=10)
```

# Example
See [sample notebook](./DenseSlim_test.ipynb)
