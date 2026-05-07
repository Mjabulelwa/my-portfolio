#Load & prepare Data
import pandas as pd

df = pd.read_csv("expenses.csv")
df["date"] = pd.to_datetime(df["date"])

# Create new features
df["month"] = df["date"].dt.month
df["day"] = df["date"].dt.day_name()

print(df.head())

#Basic Insights
#Total spending
total_spent = df["amount"].sum()
print("Total Spending: R", total_spent)

#spending by category
category_spending = df.groupby("category")["amount"].sum()
print(category_spending)

#Visualisations
import matplotlib.pyplot as plt
import seaborn as sns

# Category spending bar chart
category_spending.plot(kind="bar", title="Spending by Category")
plt.show()

# Daily spending trend
df.groupby("date")["amount"].sum().plot(title="Daily Spending Trend")
plt.show()

print("******************************Smart Insights System************************************")
#Smart Insights System
# Detect overspending category
highest_category = category_spending.idxmax()
print(f"You spend the most on: {highest_category}")

# Detect high spending days
daily_spending = df.groupby("date")["amount"].sum()
avg_spending = daily_spending.mean()

high_days = daily_spending[daily_spending > avg_spending]
print("High spending days:")
print(high_days)

print("*******************************Machine Learning****************************************")
#Predict future spending
from sklearn.linear_model import LinearRegression
import numpy as np

# Prepare data
df["day_number"] = (df["date"] - df["date"].min()).dt.days

X = df[["day_number"]]
y = df["amount"]

model = LinearRegression()
model.fit(X, y)

# Predict next 5 days
future_days = np.array(range(df["day_number"].max()+1, df["day_number"].max()+6)).reshape(-1,1)
predictions = model.predict(future_days)

print("Predicted future expenses:", predictions)