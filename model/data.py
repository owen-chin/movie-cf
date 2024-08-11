import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(movie_path, rating_path):
    movies_df = pd.read_csv(movie_path)
    ratings_df = pd.read_csv(rating_path)

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()

    return movies_df, ratings_df, user_encoder, movie_encoder
