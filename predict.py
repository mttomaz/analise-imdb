from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import pandas as pd
from utils import duration_to_minutes

df = pd.read_csv("imdb.csv")

df = df[["title", "year", "rating_imdb", "oscar", "win", "duration", "budget"]]
df = df.dropna()

df["oscar"] = df["oscar"].astype(bool).astype(int)
df["win"] = df["win"].astype(bool).astype(int)
df["duration"] = df["duration"].apply(duration_to_minutes)
df = df.dropna(subset=["duration"])

X = df[["year", "duration", "budget", "oscar", "win"]]  # escolha colunas disponíveis
y = df["rating_imdb"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Erro quadrático médio (MSE):", mse)
