import numpy as np
import pandas as pd
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

# Inputs: ratings (user_id, movie_id, rating), clusters (user_id -> cluster_id),
#         cluster_genres (cluster_id -> [top genres]), movies (movie_id -> [genres])

def get_top_movies_for_cluster(ratings_df, clusters_df, movies_df, cluster_genres_df, top_n=20, min_ratings=4):
    """
    Generate top-rated movie recommendations for each cluster, considering ratings and genre preferences.
    
    Parameters:
    - ratings_df (pd.DataFrame): Columns 'UserID', 'MovieID', 'Rating'
    - clusters_df (pd.DataFrame): Columns 'UserID', 'cluster_id'
    - movies_df (pd.DataFrame): Columns 'MovieID', 'genres'
    - cluster_genres_df (pd.DataFrame): Columns 'cluster_id', 'top_genres'
    - top_n (int): Number of top movies to return per cluster
    - min_ratings (int): Minimum number of ratings for a movie to be considered
    
    Returns:
    - dict: {cluster_id: [(movie_id, adjusted_score), ...]}
    """
    
    # Step 1: Merge ratings with cluster assignments
    ratings_with_clusters = pd.merge(ratings_df, clusters_df, on='UserID')
    
    # Step 2: Calculate average ratings and counts per movie per cluster
    cluster_movie_stats = ratings_with_clusters.groupby(['cluster_id', 'MovieID']).agg(
        avg_rating=('Rating', 'mean'),
        rating_count=('Rating', 'count')
    ).reset_index()
    
    # Step 3: Filter movies with sufficient ratings
    cluster_movie_stats = cluster_movie_stats[cluster_movie_stats['rating_count'] >= min_ratings]
    
    # Step 4: Prepare genre data for matching
    movies_df['genres_list'] = movies_df['genres'].str.split('|')
    cluster_genres_df['top_genres_list'] = cluster_genres_df['top_genres'].str.split('|')
    
    # Merge movie genres into cluster_movie_stats
    cluster_movie_stats = pd.merge(cluster_movie_stats, movies_df[['MovieID', 'genres_list']], on='MovieID')
    
    # Step 5: Check genre overlap and adjust scores
    def has_top_genre(row, cluster_genres):
        cluster_top_genres = cluster_genres.get(row['cluster_id'], [])
        return any(genre in cluster_top_genres for genre in row['genres_list'])
    
    cluster_genres_dict = cluster_genres_df.set_index('cluster_id')['top_genres_list'].to_dict()
    cluster_movie_stats['matches_top_genre'] = cluster_movie_stats.apply(
        lambda row: has_top_genre(row, cluster_genres_dict), axis=1
    )
    
    # Boost rating by 20% if genres match
    cluster_movie_stats['adjusted_score'] = cluster_movie_stats.apply(
        lambda row: row['avg_rating'] * 1.2 if row['matches_top_genre'] else row['avg_rating'], axis=1
    )
    
    # Step 6: Rank and select top movies per cluster
    top_movies = cluster_movie_stats.groupby('cluster_id').apply(
        lambda x: x.nlargest(top_n, 'adjusted_score')[['MovieID', 'adjusted_score']]
    ).reset_index()
    
    # Convert to dictionary format
    top_movies_dict = top_movies.groupby('cluster_id').apply(
        lambda x: list(zip(x['MovieID'], x['adjusted_score']))
    ).to_dict()
    
    return top_movies_dict

# Output: top_movies (cluster_id -> [(movie_id, score), ...])

###########################################################################

# Hybrid Recommender System Funcion #3

###########################################################################

def recommend_for_new_user(user_gender, user_age, user_profession, clusters_df, top_movies_dict, gender_weight=100, profession_weight=10):
    """
    Recommend movies to a new user based on their demographic information by mapping them to the most appropriate cluster.
    
    Parameters:
    - user_gender (str): Gender of the new user ('Male' or 'Female').
    - user_age (int): Age of the new user.
    - user_profession (str): Profession of the new user.
    - clusters_df (pd.DataFrame): DataFrame with columns 'ClusterID', 'Male/Female', 'Average Age', 'Profession'.
    - top_movies_dict (dict): Dictionary mapping 'ClusterID' to a list of top movies [(movie_id, score), ...].
    - gender_weight (int): Weight for gender mismatch in distance calculation (default=100).
    - profession_weight (int): Weight for profession mismatch in distance calculation (default=10).
    
    Returns:
    - list: List of recommended movie IDs.
    """
    # Calculate mismatch and difference columns
    clusters_df['gender_mismatch'] = (clusters_df['Male/Female'] != user_gender).astype(int)
    clusters_df['age_diff'] = abs(clusters_df['Average Age'] - user_age)
    clusters_df['profession_mismatch'] = (clusters_df['Profession'] != user_profession).astype(int)
    
    # Calculate distance with gender as the most crucial factor
    clusters_df['distance'] = (gender_weight * clusters_df['gender_mismatch'] +
                               clusters_df['age_diff'] +
                               profession_weight * clusters_df['profession_mismatch'])
    
    # Find the cluster with the smallest distance
    best_cluster = clusters_df.loc[clusters_df['distance'].idxmin()]
    best_cluster_id = best_cluster['ClusterID']
    
    # Get the top movies for the best cluster
    recommendations = [movie_id for movie_id, score in top_movies_dict[best_cluster_id]]
    
    return recommendations

###########################################################################

# Hybrid Recommender System Funcion #4

###########################################################################