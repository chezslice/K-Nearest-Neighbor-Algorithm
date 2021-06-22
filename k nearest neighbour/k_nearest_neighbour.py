import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

# Loading the data. Using pandas module.
data = pd.read_csv("car.data")
print(data.head())

# Converting data in order to train, we convert string into numerical values.
le = preprocessing.LabelEncoder()

# Fit the model with the proper array that contains new values.
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

# Combining the data to fit into a feature and label list. 
X = list(zip(buying, maint, door, persons, lug_boot, safety))
Y = list(cls)

# Training and testing variables are assigned to the feature and label variables.
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size = 0.1)