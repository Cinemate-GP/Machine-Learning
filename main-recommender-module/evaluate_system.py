import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from hybrid_recommender import HybridRecommender

# Add content_based_filtering directory to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'content_based_filtering'))
sys.path.append(module_path)

def load_ratings(ratings_path):
    """Load and optimize ratings dataset."""
    df = pd.read_csv(ratings_path)
    df['UserID'] = df['UserID'].astype('int32')
    df['MovieID'] = df['MovieID'].astype('int32')
    df['Rating'] = df['Rating'].astype('int8')
    return df

def evaluate_rmse(recommender, test_df, sample_size=10000):
    """Compute RMSE for rating predictions."""
    # Sample test data
    test_sample = test_df.sample(n=min(sample_size, len(test_df)), random_state=42)
    
    predictions = []
    actuals = []
    for _, row in tqdm(test_sample.iterrows(), total=len(test_sample), desc="Predicting ratings"):
        try:
            pred_rating = recommender.recommend(user_id=row['UserID'], movie_id=row['MovieID'])
            predictions.append(pred_rating)
            actuals.append(row['Rating'])
        except ValueError:
            continue
    
    if not predictions:
        return np.nan
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(actuals))**2))
    return rmse

def evaluate_precision_k(recommender, high_rated, k=5, n_users=1000):
    """Compute Precision@k for top-N recommendations."""
    # Sample users with enough high-rated movies
    user_counts = high_rated['UserID'].value_counts()
    valid_users = user_counts[user_counts >= 2].index
    test_users = np.random.choice(valid_users, size=min(n_users, len(valid_users)), replace=False)
    
    precisions = []
    for user_id in tqdm(test_users, desc="Evaluating users"):
        user_likes = high_rated[high_rated['UserID'] == user_id]['MovieID'].values
        try:
            recs_df = recommender.recommend(user_id=user_id, top_n=k)
            rec_movie_ids = recs_df['MovieID'].values
            relevant = len(set(rec_movie_ids) & set(user_likes))
            precision = relevant / k
            precisions.append(precision)
        except ValueError:
            continue
    
    return np.mean(precisions) if precisions else 0

def main():
    # Model and data paths
    SCRIPT_DIR = Path(__file__).parent.resolve()
    MODEL_DIR = SCRIPT_DIR / "models"
    RATINGS_PATH = "k:\\MachineProject\\enviro\\machine-learning-dev\\preprocessing\\ratings.csv"  # Adjust if needed
    
    # Initialize recommender
    print("Initializing HybridRecommender...")
    recommender = HybridRecommender(
        cf_model_path=str(MODEL_DIR / "cf_model.pkl"),
        cb_model_path=str(MODEL_DIR / "cb_model.joblib")
    )
    
    # Load ratings
    print("Loading ratings data...")
    ratings_df = load_ratings(RATINGS_PATH)
    
    # Split test set
    print("Splitting test set...")
    _, test_df = train_test_split(ratings_df, test_size=0.2, random_state=42)
    high_rated = test_df[test_df['Rating'] >= 4][['UserID', 'MovieID']]
    
    # Filter test_df to valid movies (optional, if errors occur)
    valid_movies = recommender.cb_model.metadata['MovieID']
    test_df = test_df[test_df['MovieID'].isin(valid_movies)]
    high_rated = high_rated[high_rated['MovieID'].isin(valid_movies)]
    
    # Evaluate RMSE
    print("Evaluating RMSE...")
    rmse = evaluate_rmse(recommender, test_df, sample_size=10000)
    print(f"RMSE: {rmse:.4f}")
    
    # Evaluate Precision@k
    k = 5
    print(f"Evaluating Precision@{k}...")
    avg_precision = evaluate_precision_k(recommender, high_rated, k=k, n_users=1000)
    print(f"Average Precision@{k}: {avg_precision:.4f}")

if __name__ == "__main__":
    main()