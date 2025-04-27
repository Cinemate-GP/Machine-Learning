from joblib import dump
import pandas as pd
import numpy as np

# Assume movies_df has movieId and genres columns
# Example: movies_df = pd.read_csv('path/to/movies.csv')
# If genres are in beegar_data, adjust accordingly
movies_df = pd.read_csv("K:\MachineProject\Data\ml-32m\movies.csv")

# Step 1: Extract unique genres
all_genres = set()
for genres in movies_df['genres']:
    all_genres.update(genres.split('|'))
all_genres = sorted(list(all_genres))  # e.g., ['Action', 'Adventure', 'Children', 'Comedy']

# Step 2: Create genre vectors
genres_dict = {}
for _, row in movies_df.iterrows():
    movie_id = row['movieId']
    movie_genres = row['genres'].split('|')
    # Binary vector: 1 if genre is present, 0 otherwise
    genre_vector = np.array([1 if genre in movie_genres else 0 for genre in all_genres])
    genres_dict[movie_id] = genre_vector

# Result: genres_dict = {'1': array([1, 0, 0, 1]), '2': array([0, 1, 1, 0])}

movie_ids = movies_df['movieId'].unique()

genres_dict = {movie_id: genre_vector for movie_id in movie_ids}  # Example genre vectors
dump(genres_dict, './genres_dict.pkl')