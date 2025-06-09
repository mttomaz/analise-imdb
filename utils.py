import pandas as pd
import re


def duration_to_minutes(text):
  if pd.isnull(text):
    return None
  hours = re.search(r"(\d+)h", text)
  minutes = re.search(r"(\d+)min", text)
  total = 0
  if hours:
    total += int(hours.group(1)) * 60
  if minutes:
    total += int(minutes.group(1))
  return total


def read_imdb(csv):
  df = pd.read_csv(csv)

  # Filtrar e limpar dados relevantes
  df = df[["title", "year", "rating_imdb", "oscar", "win", "duration", "budget"]]
  df = df.dropna()

  # Garantir tipos corretos
  df["oscar"] = df["oscar"].astype(bool).astype(int)
  df["win"] = df["win"].astype(bool).astype(int)
  df["duration"] = df["duration"].apply(duration_to_minutes)

  return df
