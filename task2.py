import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


current_folder = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(current_folder, "IMDB Movies India.csv")

df = pd.read_csv(csv_path, encoding="latin1")

print(df.head())

print(df.info())

print(df.isnull().sum())

df = df.dropna()

encoder = LabelEncoder()

categorical_columns = [
    "Genre",
    "Director",
    "Actor 1",
    "Actor 2",
    "Actor 3"
]

for col in categorical_columns:
    df[col] = encoder.fit_transform(df[col])

X = df[categorical_columns]
y = df["Rating"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)

mse = mean_squared_error(y_test, y_pred)

rmse = mse ** 0.5

r2 = r2_score(y_test, y_pred)

print("\nMean Absolute Error :", mae)
print("Mean Squared Error :", mse)
print("Root Mean Squared Error :", rmse)
print("R2 Score :", r2)

plt.figure(figsize=(6,4))
sns.histplot(df["Rating"], bins=20, kde=True)
plt.title("Movie Rating Distribution")
plt.show()

plt.figure(figsize=(10,6))
sns.countplot(y="Genre", data=df, order=df["Genre"].value_counts().index[:10])
plt.title("Top 10 Genres")
plt.show()

plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings")
plt.show()

plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

print("\nProject Completed Successfully!")