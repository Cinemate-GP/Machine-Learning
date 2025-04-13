import pandas as pd
from surprise import Dataset, Reader
from surprise.prediction_algorithms import SVD
from content_based_filtering.CBClass import ContentBasedRecommender, genre_tokenizer  # Your optimized CB class
from joblib import load, dump
from typing import Union, List

class HybridRecommender:
    def __init__(self,
                 cf_model_path: str,
                 cb_model_path: str = None,
                 cb_metadata_path: str = None):
        """
        Args:
            cf_model_path: Path to trained CF model (.pkl)
            cb_model_path: Path to saved CB model (.joblib)
            cb_metadata_path: Path to movie metadata (if CB not pre-trained)
        """
        # Load Collaborative Filtering model
        self.cf_model = load(cf_model_path)
        
        # Initialize Content-Based model
        if cb_model_path:
            self.cb_model = ContentBasedRecommender.load(cb_model_path)
        else:
            self.cb_model = ContentBasedRecommender()
            self.cb_metadata = pd.read_csv(cb_metadata_path)
            self.cb_model.fit(self.cb_metadata)

    def recommend(self,
                 user_id: int,
                 movie_id: int = None,
                 top_n: int = 5,
                 alpha: float = 0.7) -> Union[pd.DataFrame, float]:
        """
        Get hybrid recommendations.
        
        Args:
            user_id: Target user ID
            movie_id: If provided, returns score for this movie
            top_n: Number of recommendations (if movie_id=None)
            alpha: Weight for CF (0.7 = 70% CF, 30% CB)
            
        Returns:
            - If movie_id provided: Hybrid score (float)
            - Else: DataFrame of top_n recommendations
        """
        if movie_id is not None:
            # Single prediction mode
            return self._hybrid_score(user_id, movie_id, alpha)
        else:
            # Top-N recommendation mode
            return self._top_n_recommendations(user_id, top_n, alpha)

    def _hybrid_score(self, user_id: int, movie_id: int, alpha: float) -> float:
        """Calculate hybrid score for one user-movie pair."""
        # CF prediction
        cf_score = self.cf_model.predict(user_id, movie_id).est
        
        # CB similarity (normalized 0-1)
        cb_score = self._get_cb_score(movie_id)
        
        # Hybrid score
        return alpha * cf_score + (1 - alpha) * cb_score

    def _get_cb_score(self, movie_id: int) -> float:
        """Get normalized CB similarity score (0-1 scale)."""
        try:
            # Get most similar movie's similarity score
            recs = self.cb_model.recommend(movie_id, top_n=1, return_scores=True)
            return recs['similarity'].values[0]
        except (ValueError, IndexError):
            return 0.5  # Default score for cold-start items

    def _top_n_recommendations(self,
                              user_id: int,
                              top_n: int,
                              alpha: float) -> pd.DataFrame:
        """Get top-N hybrid recommendations."""
        # Get candidate movies (e.g., not rated by user)
        all_movies = self.cb_model.metadata['MovieID'].unique()
        rated_movies = self._get_rated_movies(user_id)
        candidates = list(set(all_movies) - set(rated_movies))
        
        # Score all candidates
        scores = []
        for movie_id in candidates[:1000]:  # Limit to top 1000 for speed
            scores.append((movie_id, self._hybrid_score(user_id, movie_id, alpha)))
        
        # Return top-N
        return pd.DataFrame(
            sorted(scores, key=lambda x: -x[1])[:top_n],
            columns=['MovieID', 'hybrid_score']
        ).merge(
            self.cb_model.metadata,
            on='MovieID'
        )

    def _get_rated_movies(self, user_id: int) -> List[int]:
        """Helper to get movies already rated by user."""
        # Implement based on your data storage
        # Example: Query ratings database
        return []  # Placeholder

    def save(self, path: str):
        """Save hybrid model (excluding CF model which is separate)."""
        dump({
            'cb_model': self.cb_model,
            'config': {
                'cb_metadata_path': None if hasattr(self, 'cb_metadata') else self.cb_metadata_path
            }
        }, path)

    @classmethod
    def load(cls, path: str, cf_model_path: str):
        """Load hybrid model."""
        data = load(path)
        return cls(
            cf_model_path=cf_model_path,
            cb_model_path=None,
            cb_metadata_path=data['config']['cb_metadata_path']
        )