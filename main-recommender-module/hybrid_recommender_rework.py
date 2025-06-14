import numpy as np
import pandas as pd
import faiss
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

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

def get_top_movies_for_cluster(ratings_df, clusters_df, movies_df, cluster_genres_df, top_n=20, min_ratings=5):
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
    cluster_movie_stats = ratings_with_clusters.groupby(['ClusterID', 'MovieID']).agg(
        avg_rating=('Rating', 'mean'),
        rating_count=('Rating', 'count')
    ).reset_index()
    
    # Step 3: Filter movies with sufficient ratings
    cluster_movie_stats = cluster_movie_stats[cluster_movie_stats['rating_count'] >= min_ratings]
    
    # Step 4: Prepare genre data for matching
    movies_df['genres_list'] = movies_df['Genres'].str.split('|')
    cluster_genres_df['top_genres_list'] = cluster_genres_df['Genres Ranked by Score'].str.split(',')
    
    # Merge movie genres into cluster_movie_stats
    cluster_movie_stats = pd.merge(cluster_movie_stats, movies_df[['MovieID', 'genres_list']], on='MovieID')
    
    # Step 5: Check genre overlap and adjust scores
    def has_top_genre(row, cluster_genres):
        cluster_top_genres = cluster_genres.get(row['ClusterID'], [])
        return any(genre in cluster_top_genres for genre in row['genres_list'])
    
    cluster_genres_dict = cluster_genres_df.set_index('ClusterID')['top_genres_list'].to_dict()
    cluster_genres_dict.pop(-1, None)  # Remove the entry for ClusterID -1
    cluster_movie_stats['matches_top_genre'] = cluster_movie_stats.apply(
        lambda row: has_top_genre(row, cluster_genres_dict), axis=1
    )
    
    # Boost rating by 20% if genres match
    cluster_movie_stats['adjusted_score'] = cluster_movie_stats.apply(
        lambda row: row['avg_rating'] * 1.2 if row['matches_top_genre'] else row['avg_rating'], axis=1
    )
    
    # Step 6: Rank and select top movies per cluster
    top_movies = cluster_movie_stats.groupby('ClusterID').apply(
        lambda x: x.nlargest(top_n, 'adjusted_score')[['MovieID', 'adjusted_score']]
    ).reset_index()
    
    noiseIndexes = top_movies[top_movies['ClusterID'] == -1].index
    top_movies = top_movies.drop(noiseIndexes).reset_index(drop=True)
    top_movies['adjusted_score'] = top_movies.adjusted_score.round(3)

    # Convert to dictionary format
    top_movies_dict = top_movies.groupby('ClusterID').apply(
        lambda x: list(zip(x['MovieID'], x['adjusted_score']))
    ).to_dict()
    
    return top_movies_dict

# Output: top_movies (cluster_id -> [(movie_id, score), ...])

###########################################################################

# Hybrid Recommender System Funcion #3

###########################################################################

def parse_distribution(dist_str):
    """Parse distribution string into proportions (e.g., 'M:168, F:72' -> {'M': 0.7, 'F': 0.3})."""
    counts = {k.strip(): int(v) for k, v in (pair.split(':') for pair in dist_str.split(','))}
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}

def recommend_for_new_user(user_gender, user_age, user_profession, clusters_df, top_movies_dict):
    """
    Assign a new user to a cluster and recommend movies using bias-based scoring.

    Parameters:
    - user_gender (str): 'Male' or 'Female'.
    - user_age (int): User's age.
    - user_profession (str): User's profession.
    - clusters_df (pd.DataFrame): Cluster data with distribution columns.
    - top_movies_dict (dict): {cluster_id: [(movie_id, score), ...]}.

    Returns:
    - list: Recommended movie IDs.
    """

    # Define age groups
    age_groups = {
        '18-25': (18, 25),
        '26-35': (26, 35),
        '36-45': (36, 45),
        '46-55': (46, 55),
        '56+': (56, 100)
    }

    #user_age_group = next(group for group, (low, high) in age_groups.items() if low <= user_age <= high)

    # Parse distributions into proportions
    clusters_df['gender_props'] = clusters_df['Male-Female Distribution'].apply(parse_distribution)
    #clusters_df['age_props'] = clusters_df['Age Group Distribution'].apply(parse_distribution)
    clusters_df['occupation_props'] = clusters_df['Occupation Ranking by Number'].apply(parse_distribution)

    # Calculate biases
    clusters_df['gender_bias'] = clusters_df['gender_props'].apply(
        lambda props: props.get(user_gender[0], 0) - 0.5  # M or F, baseline 50%
    )
    #clusters_df['age_proportion'] = clusters_df['age_props'].apply(
    #    lambda props: props.get(user_age_group, 0)
    #)
    clusters_df['profession_bias'] = clusters_df['occupation_props'].apply(
        lambda props: props.get(user_profession, 0) - 0.2  # Baseline 20% assuming 20 professions
    )

    # Compute final score: Adjust age proportion with gender bias, add weighted profession bias
    profession_weight = 0.2  # Profession has less influence
    clusters_df['score'] = ((1 + clusters_df['gender_bias'])) + \
                            (profession_weight * clusters_df['profession_bias'])
    
    # Select cluster with highest score
    best_cluster = clusters_df.loc[clusters_df['score'].idxmax()]
    best_cluster_id = best_cluster['ClusterID']
    
    # Get the top movies for the best cluster
    preliminary_recommendations = [movie_id for movie_id, score in top_movies_dict[best_cluster_id]]
    
    return preliminary_recommendations

###########################################################################

# Hybrid Recommender System Funcion #4

###########################################################################

# Load movie data
movies_df = pd.read_csv('ml_data/movies_1m.csv')

# Load cluster assignments (assumed to exist from prior clustering)
clusters_df = pd.read_csv('ml_data/movie_clusters.csv')  # Columns: MovieID, cluster

# Merge data
movie_data = pd.merge(movies_df, clusters_df, on='MovieID')

# Extract and binarize genres
mlb = MultiLabelBinarizer()
movie_data['genres_list'] = movie_data['Genres'].str.split('|')
genre_vectors = mlb.fit_transform(movie_data['genres_list'])
movie_data = pd.concat([movie_data, pd.DataFrame(genre_vectors, columns=mlb.classes_)], axis=1)

# Compute cluster centroids
cluster_centroids = movie_data.groupby('cluster')[mlb.classes_].mean()

# Prepare FAISS index for ANN search
# Normalize vectors for cosine similarity (FAISS uses inner product)
genre_vectors = genre_vectors.astype(np.float32)
norms = np.linalg.norm(genre_vectors, axis=1, keepdims=True)
norms[norms == 0] = 1  # Avoid division by zero
normalized_vectors = genre_vectors / norms
index = faiss.IndexFlatIP(normalized_vectors.shape[1])  # Inner product index
index.add(normalized_vectors)  # Add vectors to index

def assign_new_movie_to_cluster(new_movie_genres):
    """
    Assign a new movie to a cluster based on genre similarity to centroids.
    
    Args:
        new_movie_genres (list): List of genres (e.g., ['Action', 'Comedy'])
    
    Returns:
        int: Assigned cluster ID
    """
    # Create genre vector for the new movie
    new_movie_vec = mlb.transform([new_movie_genres])[0].reshape(1, -1)
    
    # Normalize for cosine similarity
    norm = np.linalg.norm(new_movie_vec)
    if norm == 0:
        norm = 1
    new_movie_vec_normalized = new_movie_vec / norm
    
    # Compute similarity to each cluster centroid
    similarities = np.dot(new_movie_vec_normalized, cluster_centroids.values.T)[0]
    
    # Assign to the cluster with the highest similarity
    assigned_cluster = cluster_centroids.index[np.argmax(similarities)]
    return assigned_cluster


def find_top_similar_movies_with_ann(new_movie_genres, top_n=5):
    """
    Find the top N similar movies using FAISS ANN search.
    
    Args:
        new_movie_genres (list): List of genres (e.g., ['Action', 'Comedy'])
        top_n (int): Number of similar movies to return
    
    Returns:
        list: List of (MovieID, similarity_score) tuples
    """
    # Create genre vector for the new movie
    new_movie_vec = mlb.transform([new_movie_genres])[0].astype(np.float32)
    
    # Normalize for cosine similarity
    norm = np.linalg.norm(new_movie_vec)
    if norm == 0:
        norm = 1
    new_movie_vec_normalized = (new_movie_vec / norm).reshape(1, -1)
    
    # Search FAISS index
    similarities, indices = index.search(new_movie_vec_normalized, top_n)
    
    # Retrieve MovieIDs and similarities
    movie_ids = movie_data['MovieID'].values
    top_similar = [(movie_ids[i], similarities[0][j]) for j, i in enumerate(indices[0])]
    
    return top_similar