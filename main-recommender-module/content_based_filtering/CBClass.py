from joblib import dump, load
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD
from tqdm import tqdm

# Add this at the TOP of your content_based.py file
def genre_tokenizer(x):
    """Tokenize pipe-separated genre strings"""
    return x.split(',')

class ContentBasedRecommender:
    def __init__(self, 
                 vectorizer_params={'tokenizer': genre_tokenizer, 'dtype': np.float32},
                 n_components=28,
                 n_neighbors=10,
                 metric='cosine'):
        """
        Args:
            vectorizer_params: Parameters for TfidfVectorizer
            n_components: Dimensions for TruncatedSVD (set to None to disable)
            n_neighbors: Number of similar items to retrieve
            metric: Distance metric ('cosine', 'euclidean', etc.)
        """
        self.vectorizer = TfidfVectorizer(**vectorizer_params)
        self.n_components = n_components
        self.svd = TruncatedSVD(n_components=n_components) if n_components else None
        self.nn_model = NearestNeighbors(n_neighbors=n_neighbors, metric=metric)
        self.metadata = None
        self.is_fitted = False

    def fit(self, metadata, genre_col='Genres'):
        """
        Train the model on movie metadata.
        
        Args:
            metadata: DataFrame with movie_id and genres
            genre_col: Name of the column containing genre strings
        """
        self.metadata = metadata.reset_index(drop=True)
        
        # Vectorize genres
        genre_vectors = self.vectorizer.fit_transform(metadata[genre_col])
        
        # Dimensionality reduction
        if self.svd:
            genre_vectors = self.svd.fit_transform(genre_vectors)
        
        # Build nearest neighbors index
        self.nn_model.fit(genre_vectors)
        self.is_fitted = True
        return self

    def recommend(self, movie_id, top_n=5, return_scores=False):
        """
        Get similar movies for a given movie_id.
        
        Args:
            movie_id: ID of the query movie
            top_n: Number of recommendations
            return_scores: Whether to include similarity scores
            
        Returns:
            DataFrame with recommended movies (and scores if return_scores=True)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
            
        try:
            idx = self.metadata.index[self.metadata['MovieID'] == movie_id][0]
        except IndexError:
            raise ValueError(f"movie_id {movie_id} not found in metadata")
        
        # Get vector for the query movie
        query_genres = self.metadata.iloc[idx]['Genres']
        query_vec = self.vectorizer.transform([query_genres])
        if self.svd:
            query_vec = self.svd.transform(query_vec)
        
        # Find nearest neighbors
        distances, indices = self.nn_model.kneighbors(query_vec, n_neighbors=top_n+1)
        
        # Exclude the query movie itself
        results = self.metadata.iloc[indices[0][1:top_n+1]].copy()
        
        if return_scores:
            results['similarity'] = 1 - distances[0][1:top_n+1]  # Convert distance to similarity
            
        return results

    def batch_recommend(self, movie_ids, top_n=5, batch_size=1000):
        """Recommend for multiple movies in batches to save memory."""
        results = []
        for i in tqdm(range(0, len(movie_ids), batch_size)):
            batch = movie_ids[i:i+batch_size]
            results.extend([self.recommend(mid, top_n) for mid in batch])
        return pd.concat(results)
#####################################################################################################

    def save(self, filename):
        """Save the trained model to disk."""
        dump({
            'vectorizer': self.vectorizer,
            'svd': self.svd,
            'nn_model': self.nn_model,
            'metadata': self.metadata,
            'config': {
                'n_components': self.n_components,
                'is_fitted': self.is_fitted
            }
        }, filename)

    @classmethod
    def load(cls, filename):
        """Load a trained model from disk."""
        data = load(filename)
        recommender = cls(
            vectorizer_params=data['vectorizer'].get_params(),
            n_components=data['config']['n_components'],
            n_neighbors=data['nn_model'].n_neighbors,
            metric=data['nn_model'].metric
        )
        recommender.vectorizer = data['vectorizer']
        recommender.svd = data['svd']
        recommender.nn_model = data['nn_model']
        recommender.metadata = data['metadata']
        recommender.is_fitted = data['config']['is_fitted']
        return recommender