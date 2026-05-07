#Import libraries 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #Seaborn is a Python data visualization library built on top of Matplotlib. 
#It makes it easier to create beautiful, informative statistical graphics.

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

#Hyperparameter Tuning
from sklearn.model_selection import GridSearchCV

#Load and Inspect Data
df = pd.read_csv("patient_visits.csv")

rows = len(df)
print(df.head(rows))
print(df.info())
print(df.isnull().sum())

#Data Cleaning
# Convert date column
df["visit_date"] = pd.to_datetime(df["visit_date"], errors='coerce')
# Handle missing values
df = df.dropna()
# Remove duplicates
df = df.drop_duplicates()

#Feature Engineering
# Extract useful features from date
df["visit_day"] = df["visit_date"].dt.day
df["visit_month"] = df["visit_date"].dt.month
df["visit_weekday"] = df["visit_date"].dt.weekday

# Encode categorical variables
le = LabelEncoder()

for col in ["gender", "diagnosis", "visit_type", "doctor_id"]:
    df[col] = le.fit_transform(df[col])

#Exploratory Data Analysis (EDA)
# Visits per month
sns.countplot(x="visit_month", data=df)
plt.title("Patient Visits per Month")
plt.show()

# Readmission distribution
sns.countplot(x="readmitted", data=df)
plt.title("Readmission Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()

#Machine Learning Model
X = df.drop(["patient_id", "visit_date", "readmitted"], axis=1)
y = df["readmitted"]

#Split Data
X_train, X_test, y_train, y_test = train_test_split(
 X, y, test_size=0.2, random_state=42
) 

#Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

#Module Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

#Feature Importance
importances = model.feature_importances_
features = X.columns

feat_imp = pd.Series(importances, index=features).sort_values()

feat_imp.plot(kind="barh")
plt.title("Feature Importance")
plt.show()


""" Insights You Can Generate
Which patients are high-risk for readmission
Peak visit times (days/months)
Impact of:
wait time
diagnosis
visit type
Doctor workload trends """

#11. Optional Improvements (Real-World Level)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier  #It implements gradient boosting, 
#an ensemble technique that builds many decision trees sequentially, 
# where each new tree corrects the errors of the previous ones.

#Save Model
import joblib
joblib.dump(model, "patient_model.pkl")