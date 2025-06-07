import pandas as pd
df = pd.read_csv('social_media_posts_10k.csv')
print("Duplicate rows:", df.duplicated().sum())