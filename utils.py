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
