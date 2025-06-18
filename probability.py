import matplotlib.pyplot as plt
from utils import read_imdb

df = read_imdb("imdb.csv")

# Amostragem Aleatória Simples (AAS)
amostra = df.sample(n=2000, random_state=42)

# Estatísticas comparativas
media_pop = df["rating_imdb"].mean()
media_amostra = amostra["rating_imdb"].mean()

print("=== Comparação de Médias ===")
print("Média IMDb - População:", round(media_pop, 2))
print("Média IMDb - Amostra:", round(media_amostra, 2))

plt.hist(df["rating_imdb"], bins=20, alpha=0.5, label="População")
plt.hist(amostra["rating_imdb"], bins=20, alpha=0.5, label="Amostra")
plt.title("Distribuição das notas IMDb: População vs Amostra")
plt.xlabel("Nota IMDb")
plt.ylabel("Frequência")
plt.legend()
plt.show()
