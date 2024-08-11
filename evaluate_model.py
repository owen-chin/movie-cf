import torch
import pandas as pd
from torch.utils.data import DataLoader
from model.model import RecSysModel
from model.dataset import MovieDataset
from model.data import load_and_preprocess_data
from sklearn.metrics import mean_squared_error
from collections import defaultdict
import numpy as np

#define file paths
movie_path = "data/ml-latest-small/movies.csv"
rating_path = "data/ml-latest-small/ratings.csv"
test_ratings_path = "data/ml-latest-small/test_ratings.csv"
MODEL_PATH = "./models/saved_model.pth"
BATCH_SIZE = 32

# Load test data
df_valid = pd.read_csv(test_ratings_path)

# Load and preprocess data
(
    movies_df, 
    ratings_df,
    user_encoder, 
    movie_encoder
) = load_and_preprocess_data(
    movie_path,
    rating_path
)

valid_dataset = MovieDataset(
    users=df_valid.userId.values,
    movies=df_valid.movieId.values,
    ratings=df_valid.rating.values
)

valid_loader = DataLoader(dataset=valid_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=False
)

# Initialize device and model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
recommendation_model = RecSysModel(
    n_users=len(user_encoder.classes_),
    n_movies=len(movie_encoder.classes_),
    embedding_size=64,
    hidden_dim=128,
    dropout_rate=0.1
)
recommendation_model.load_state_dict(torch.load(MODEL_PATH))
recommendation_model.to(device)

def calculate_precision_recall(user_ratings, k, threshold):
    user_ratings.sort(key=lambda x: x[0], reverse=True)
    n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
    n_rec_k = sum(est >= threshold for est, _ in user_ratings[:k])
    n_rel_and_rec_k = sum(
        (true_r >= threshold) and (est >= threshold) for est, true_r in user_ratings[:k]
    )

    precision = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 1
    recall = n_rel_and_rec_k / n_rel if n_rel != 0 else 1
    return precision, recall

def find_RMSE():
    # Root Mean Squared Error
    y_true = []
    y_pred = []

    recommendation_model.eval()

    with torch.no_grad():
        for valid_data in valid_loader:
            model_output = recommendation_model(
                valid_data['users'].to(device),
                valid_data['movies'].to(device)
            )

            ratings = valid_data['ratings'].to(device)
            y_true.extend(ratings.cpu().numpy())
            y_pred.extend(model_output.cpu().numpy())

    # Calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"RMSE: {rmse:.4f}")

def main():
    find_RMSE()

    user_ratings_comparison = defaultdict(list)

    with torch.no_grad():
        for valid_data in valid_loader:
            users = valid_data["users"].to(device)
            movies = valid_data["movies"].to(device)
            ratings = valid_data["ratings"].to(device)
            output = recommendation_model(users, movies)

            for user, pred, true in zip(users, output, ratings):
                user_ratings_comparison[user.item()].append((pred.item(), true.item()))

    user_precisions = dict()
    user_based_recalls = dict()

    k = 50
    threshold = 3

    for user_id, user_ratings in user_ratings_comparison.items():
        precision, recall = calculate_precision_recall(user_ratings, k, threshold)
        user_precisions[user_id] = precision
        user_based_recalls[user_id] = recall

    average_precision = sum(prec for prec in user_precisions.values()) / len(user_precisions)
    average_recall = sum(rec for rec in user_based_recalls.values()) / len(user_based_recalls)

    print(f"Precision @ {k}: {average_precision:.4f}")
    print(f"Recall @ {k}: {average_recall:.4f}")

if __name__ == "__main__":
    main()
