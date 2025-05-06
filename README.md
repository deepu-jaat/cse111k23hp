# python project 
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score 
file_path "dataset_spotify.csv"
df pd.read_csv(file_path)
df.head()
df df.drop(columns=["Unnamed: 0", "track_id", "album_name"])
columns_to_drop ["track_name", "artists"]
df.drop(columns=columns_to_drop, inplace=True)
df.head()
print("Missing values per column:\n", df.isna().sum())
track_genre
dtype: int64

# Distribution of popularity
plt.figure(figsize=(8, 6))
sns.histplot(df['popularity'], bins 30, color='skyblue')
plt.title("Distribution of Song Popularity")
plt.xlabel("Popularity")
plt.ylabel("Frequency")
plt.show()

# Popularity vs Danceability
plt.figure(figsize=(8, 6))
sns.scatterplot(x='danceability', y='popularity', data=df, hue='explicit', alpha=0.7)
plt.title("Danceability vs Popularity")
plt.xlabel("Danceability")
plt.ylabel("Popularity")
plt.legend(title="Explicit")
plt.show()

#Boxplot of Popularity by Genre
plt.figure(figsize=(12, 6))
sns.boxplot(x='track_genre', y='popularity', data=df)
plt.title("Popularity by Genre")
plt.xlabel("Track Genre (Encoded)")
plt.ylabel("Popularity")
plt.xticks(rotation=90)
plt.show()

# Pie chart of explicit vs. non-explicit songs
explicit_counts df['explicit'].value_counts()
labels ['Non-Explicit', 'Explicit']
colors ['#66b3ff', '#ff9999']
plt.figure(figsize=(6, 6))
plt.pie(explicit_counts, labels labels, autopct='%1.1f%%', startangle=140, colors colors)
plt.title("Distribution of Explicit vs Non-Explicit Songs")
plt.axis('equal')
plt.show()
plt.figure(figsize=(8, 6))

sns.scatterplot(x= 'acousticness', y='loudness', data df, hue='explicit', alpha=0.6)
plt.title("Acousticness vs Loudness")
plt.xlabel("Acousticness")
plt.ylabel("Loudness (dB)")
plt.legend(title="Explicit")
plt.show()
plt.figure(figsize=(14,6))

popularity'].mean().sort_values(ascending=False)

genre_popularity df.groupby('track_genre') [' sns.barplot(x=genre popularity.index, y=genre popularity.values, palette='viridis')
plt.title("Average Popularity by Genre")

plt.xlabel("Track Genre")
plt.ylabel("Average Popularity")
plt.xticks(rotation-90)
plt.show()
sns.barplot(x-genre_popularity.index, y-genre_popularity.values, palette='viridis')
plt.figure(figsize-(12, 8))

numeric_df df.select_dtypes (include-[np.number]) sns.heatmap(numeric_df.corr(), annot True, cmap-"coolwarm", fnt=".2f")

plt.title("Feature Correlation Heatmap")

plt.show()