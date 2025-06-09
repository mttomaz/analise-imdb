import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

df = pd.read_csv("imdb.csv")

# Filtrar e limpar dados relevantes
df = df[["title", "year", "rating_imdb", "oscar", "win", "duration", "budget"]]
df = df.dropna()

# Garantir tipos corretos
df["oscar"] = df["oscar"].astype(bool).astype(int)
df["win"] = df["win"].astype(bool).astype(int)

# Análise exploratória rápida
sns.lineplot(data=df, x="year", y="rating_imdb")
plt.title("Tendência de notas IMDb ao longo do tempo")
plt.show()

# Regressão simples: rating ~ year
model_simple = smf.ols("rating_imdb ~ year", data=df).fit()
print("\nModelo simples: rating_imdb ~ year")
print(model_simple.summary())

# Regressão com interação: rating ~ year * oscar
model_interaction = smf.ols("rating_imdb ~ year * oscar", data=df).fit()
print("\nModelo com interação: rating_imdb ~ year * oscar")
print(model_interaction.summary())

coef = model_interaction.params
print(f"""
Interpretação:
- Coef. year: {coef["year"]:.4f} → impacto do tempo em filmes SEM oscar
- Coef. oscar: {coef["oscar"]:.4f} → diferença de nota para filmes COM oscar (ano base)
- Coef. year:oscar: {coef["year:oscar"]:.4f} → se for negativo, notas de filmes COM oscar caem mais rápido
""")

df["rating_imdb"].hist(bins=20)
plt.title("Histograma das notas IMDb")
plt.xlabel("Nota")
plt.ylabel("Frequência")
plt.show()

print("Média:", df["rating_imdb"].mean())
print("Moda:", df["rating_imdb"].mode()[0])
print("Mediana:", df["rating_imdb"].median())
print("Desvio padrão:", df["rating_imdb"].std())
