import os
import sys
from pathlib import Path
from hybrid_recommender import HybridRecommender

# Add content_based_filtering directory to sys.path
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'content_based_filtering'))
sys.path.append(module_path)

# 1. Get the correct model paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MODEL_DIR = SCRIPT_DIR / "models"

# 2. Initialize with verified paths
recommender = HybridRecommender(
    cf_model_path=str(MODEL_DIR / "cf_model.pkl"),
    cb_model_path=str(MODEL_DIR / "cb_model.joblib")
)

# 3. Test
print("Testing hybrid recommendation...")
print(recommender.recommend(user_id=1, movie_id=1193))  # Single prediction
print(recommender.recommend(user_id=1, top_n=5))       # Top-N recommendations