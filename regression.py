import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from utils import read_imdb
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

df = read_imdb("imdb.csv")

# -----------------------
# REGRESSÃO LINEAR
# -----------------------
print("\n=== REGRESSÃO LINEAR: Notas IMDb ao longo do tempo ===")
model_linear = smf.ols("rating_imdb ~ year", data=df).fit()
print(model_linear.summary())

sns.lmplot(data=df, x="year", y="rating_imdb", line_kws={"color": "red"})
plt.title("Tendência das notas IMDb com os anos")
plt.show()

# -----------------------
# REGRESSÃO LOGÍSTICA
# -----------------------
print("\n=== REGRESSÃO LOGÍSTICA: Previsão de Oscar ===")

# Variáveis preditoras e alvo
X = df[["rating_imdb", "duration", "win", "budget"]]
y = df["oscar"]

# Balanceamento de classes (opcional: undersample)
# Aqui só para teste, podemos igualar número de 0s e 1s:
df_class0 = df[df["oscar"] == 0].sample(n=df["oscar"].sum(), random_state=42)
df_class1 = df[df["oscar"] == 1]
df_balanced = pd.concat([df_class0, df_class1])

X = df_balanced[["rating_imdb", "duration", "win", "budget"]]
y = df_balanced["oscar"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model_logistic = LogisticRegression(max_iter=1000)
model_logistic.fit(X_train, y_train)

y_pred = model_logistic.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------
# INTERPRETAÇÃO
# -----------------------
print("""
Interpretação:
- A regressão linear mostra como as notas IMDb evoluem com os anos.
- A regressão logística estima a probabilidade de um filme ganhar um Oscar com base em características como nota, duração, prêmios e orçamento.
- Após balancear as classes, o modelo foi capaz de prever ganhadores e não-ganhadores com mais equilíbrio.
""")
