
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#print("all installed successully")

import pickle

data = pd.read_csv(r"C:\EXCEL DATA\Crop_recommendation.csv")


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

print(data.head())

x = data.iloc[:, :-1] #feature
y = data.iloc[:, -1] #label

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

pickle.dump(model, open("model.pkl", "wb"))
print("model saved successfully")

