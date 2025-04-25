import numpy as np
import pandas as pd
import surprise
from surprise import Dataset, Reader
from surprise.model_selection import train_test_split
from joblib import load
from sklearn.metrics.pairwise import cosine_similarity

###########################################################################

# Hybrid Recommender System Funcion #1

###########################################################################

def get_user_genre_profile(user_id, dataset, genres_dict):
    """
    Compute a user's genre preference vector based on their rated movies.
    Args:
        user_id: User ID (str or int).
        dataset: DataFrame with UserID, MovieID, Rating columns.
        genres_dict: Dict mapping MovieID to genre vector (e.g., binary or TF-IDF).
    Returns:
        np.array: User's genre preference vector.
    """
    print(f"Generating user profile for user {user_id}...")
    user_ratings = dataset[dataset['UserID'] == user_id]
    if user_ratings.empty:
        return np.zeros(len(next(iter(genres_dict.values()))))  # Default for cold-start
    # Weighted average of genre vectors by rating
    genre_vectors = np.array([genres_dict.get(mid, np.zeros_like(next(iter(genres_dict.values())))) 
                             for mid in user_ratings['MovieID']])
    weights = user_ratings['Rating'].values / user_ratings['Rating'].sum()
    return np.average(genre_vectors, axis=0, weights=weights)

def get_top_n_hybrid_recommendations(user_id, cf_model, dataset, genres_dict, k=200, n=25):
    """
    Generate top-N hybrid recommendations using CF pre-filtering and CB ranking.
    Args:
        user_id: User ID (str or int).
        cf_model: Trained Surprise SVD model.
        dataset: DataFrame with UserID, MovieID, Rating columns.
        genres_dict: Dict mapping MovieID to genre vector.
        k: Number of CF candidates to pre-filter.
        n: Number of final recommendations.
    Returns:
        list: Top-N MovieIDs.
    """
    # Step 1: CF Pre-Filtering
    print(f"Generating CF recommendations for user {user_id}...")
    all_movie_ids = dataset['MovieID'].unique()
    cf_predictions = [cf_model.predict(user_id, mid) for mid in all_movie_ids]
    cf_scores = [(pred.iid, pred.est) for pred in cf_predictions]
    cf_scores.sort(key=lambda x: x[1], reverse=True)
    top_k_movies = [movie_id for movie_id, _ in cf_scores[:k]]

    # Step 2: CB Ranking
    print(f"Generating CB recommendations for user {user_id}...")
    user_profile = get_user_genre_profile(user_id, dataset, genres_dict)
    cb_scores = []
    for movie_id in top_k_movies:
        movie_vector = genres_dict.get(movie_id, np.zeros_like(user_profile))
        if np.any(user_profile) and np.any(movie_vector):  # Avoid zero vectors
            similarity = cosine_similarity([user_profile], [movie_vector])[0][0]
        else:
            similarity = 0.0  # Fallback for no overlap
        cb_scores.append((movie_id, similarity))
    
    # Step 3: Sort and Select Top-N
    print(f"Success! Sorting final recommendation list...")
    cb_scores.sort(key=lambda x: x[1], reverse=True)
    top_n_movies = [movie_id for movie_id, _ in cb_scores[:n]]
    
    return top_n_movies

###########################################################################

# Hybrid Recommender System Funcion #2

###########################################################################