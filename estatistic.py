import matplotlib.pyplot as plt
from utils import read_imdb

df = read_imdb("imdb.csv")

# Histograma 1: Notas IMDb
df["rating_imdb"].hist(bins=20, alpha=0.7)
plt.title("Histograma das notas IMDb")
plt.xlabel("Nota")
plt.ylabel("Frequência")
plt.show()

# Histograma 2: Duração dos filmes
df["duration"].hist(bins=20, color="orange", alpha=0.7)
plt.title("Histograma da duração dos filmes")
plt.xlabel("Duração (minutos)")
plt.ylabel("Frequência")
plt.show()

# Medidas de tendência central e dispersão
print("=== Estatísticas Descritivas ===")
print("Notas IMDb:")
print("  Média:", round(df["rating_imdb"].mean(), 2))
print("  Moda:", df["rating_imdb"].mode()[0])
print("  Mediana:", df["rating_imdb"].median())
print("  Desvio padrão:", round(df["rating_imdb"].std(), 2))

print("\nDuração dos filmes:")
print("  Média:", round(df["duration"].mean(), 2))
print("  Moda:", df["duration"].mode()[0])
print("  Mediana:", df["duration"].median())
print("  Desvio padrão:", round(df["duration"].std(), 2))
