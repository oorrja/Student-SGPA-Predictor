import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score 
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

# Load the csv file
df = pd.read_excel('testing set.xlsx')



print(df.head())

# Data Preparation
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

# Select independent and dependent variable
X = df[["Sum of Internal Marks ", "10th Percentage", "12th Percentage", "Total Times Debarred","Stream","Total Subjects","Gender","Study Hours","10th Board","12th Board","Co-Curricular Activity","Library Books","Assignments","Mentor","Place of Living","Distance from College","Mode of Transport ","Family Income","Daily Screen Time","Chronic Illness","Close Friends"]]
y = df["SGPA"]


# Split the dataset into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)



# Instantiate the model
lr = LinearRegression()
rf = RandomForestRegressor(n_estimators=40)
dt = DecisionTreeRegressor()
gb = GradientBoostingRegressor()


classifier = DecisionTreeRegressor()

# Fit the model
classifier.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(classifier, open("model.pkl", "wb"))