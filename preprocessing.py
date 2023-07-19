import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('Dataset/Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# replacing missing value of the colour with columns mean
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

# replacing column with string values with Encoded value using onehotencode
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# replacing column with encoded value using label encoder
le = LabelEncoder()
y = le.fit_transform(y)

# splitting training and test samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# feature scaling  the data by standardization
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
