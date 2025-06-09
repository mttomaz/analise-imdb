from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from utils import read_imdb

df = read_imdb("imdb.csv")

X = df[["year", "duration", "budget", "oscar", "win"]]
y = df["rating_imdb"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print("Erro quadrático médio (MSE):", mse)
