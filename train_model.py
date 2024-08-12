import pandas as pd
import torch
from sklearn import model_selection
from torch.utils.data import DataLoader
from model.model import RecSysModel
from model.train import train
from model.dataset import MovieDataset
from model.data import load_and_preprocess_data
import os

print("current working directory:", os.getcwd())
script_dir = os.path.dirname(os.path.realpath(__file__))

# Construct the absolute paths
movie_path = os.path.join(script_dir, "data/ml-latest-small/movies.csv")
rating_path = os.path.join(script_dir, "data/ml-latest-small/ratings.csv")
test_ratings_path = os.path.join(script_dir, "data/ml-latest-small/test_ratings.csv")

MODEL_PATH = os.path.join(script_dir, "models/saved_model.pth")
BATCH_SIZE = 32
EPOCHS = 2

def main():
    (
        movies_df, 
        ratings_df, 
        user_encoder, 
        movie_encoder
    ) = load_and_preprocess_data(
        movie_path,
        rating_path
    )

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    n_users = ratings_df.userId.nunique()
    n_movies = ratings_df.movieId.nunique()

    ratings_df.userId = user_encoder.fit_transform(ratings_df.userId.values)
    ratings_df.movieId = movie_encoder.fit_transform(ratings_df.movieId.values)

    df_train, df_valid = model_selection.train_test_split(
        ratings_df, test_size=0.1, random_state=3, stratify=ratings_df.rating.values
    )

    df_valid.to_csv(test_ratings_path, index=False)

    train_dataset = MovieDataset(
        users = df_train.userId.values,
        movies = df_train.movieId.values,
        ratings = df_train.rating.values
    )

    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True
    )

    recommendation_model = RecSysModel(n_users, n_movies, embedding_size=64, hidden_dim=128, dropout_rate=0.1)

    optimizer = torch.optim.Adam(recommendation_model.parameters()) #gradient descent
    sch = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.7)
    loss_fn = torch.nn.MSELoss()

    train(recommendation_model.to(device),
        train_dataset, 
        train_loader, 
        loss_fn, 
        optimizer, 
        model_path=MODEL_PATH, 
        device=device,
        epochs=EPOCHS)
    

if __name__ == "__main__":
    main()
