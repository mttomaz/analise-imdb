import seaborn as sns
import matplotlib.pyplot as plt
from utils import read_imdb

df = read_imdb("imdb.csv")

# Gráfico 1: tendência da nota ao longo do tempo
sns.lineplot(data=df, x="year", y="rating_imdb")
plt.title("Tendência das notas IMDb ao longo dos anos")
plt.xlabel("Ano")
plt.ylabel("Nota IMDb")
plt.grid(True)
plt.tight_layout()
plt.show()

# Gráfico 2: dispersão entre nota e duração do filme
sns.scatterplot(data=df, x="duration", y="rating_imdb")
plt.title("Relação entre duração e nota IMDb")
plt.xlabel("Duração (minutos)")
plt.ylabel("Nota IMDb")
plt.grid(True)
plt.tight_layout()
plt.show()
