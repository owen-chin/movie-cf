# app.py
from flask import Flask, request, render_template
import pandas as pd
import torch
from model.data import load_and_preprocess_data
from model.model import load_model
from model.model import RecSysModel

app = Flask(__name__)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Load and preprocess data
(
    movies_df, 
    ratings_df, 
    user_encoder, 
    movie_encoder
) = load_and_preprocess_data(
    "data/ml-latest-small/movies.csv", 
    "data/ml-latest-small/ratings.csv"
)

# Initialize model
n_users = ratings_df.userId.nunique()
n_movies = ratings_df.movieId.nunique()

model_path = 'models/saved_model.pth'  # Ensure this path is correct and matches the path in train_model.py
recommendation_model = load_model(model_path, device, n_users, n_movies)

# Label encoding for movie ids
lbl_movie = movie_encoder

# All movies
all_movies = ratings_df['movieId'].unique().tolist()

# Function to get top recommendations
def top_recommendations(user_id, all_movies, k=5, batch_size=100):
    recommendation_model.eval()
    watched_movies = set(ratings_df[ratings_df['userId'] == user_id]['movieId'].tolist())
    unwatched_movies = [m for m in all_movies if m not in watched_movies]
    prediction = []
    top_k_recommendations = []

    with torch.no_grad():
        for i in range(0, len(unwatched_movies), batch_size):
            batched_unwatched = unwatched_movies[i:i+batch_size]
            movie_tensor = torch.tensor(batched_unwatched).to(device)
            user_tensor = torch.tensor([user_id] * len(batched_unwatched)).to(device)
            prediction_model = recommendation_model(user_tensor, movie_tensor).view(-1).tolist()
            prediction.extend(zip(batched_unwatched, prediction_model))

    prediction.sort(key=lambda x: x[1], reverse=True)
    for (m_id, _) in prediction[:k]:
        top_k_recommendations.append(m_id)

    top_k_recommendations = lbl_movie.inverse_transform(top_k_recommendations)
    return top_k_recommendations

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        user_id = int(request.form['user_id'])
        if user_id not in ratings_df["userId"].values:
            raise ValueError(f'User ID {user_id} does not exist')
        recommendations = top_recommendations(user_id, all_movies, k=5)
        recommendations_titles = [movies_df[movies_df['movieId'] == movie_id]['title'].values[0] for movie_id in recommendations]
        return render_template('index.html', recommendations_titles=recommendations_titles, recommendations=recommendations)
    except ValueError as e:
        app.logger.error(f"Error: {e}")

        return render_template('index.html', error=str(e))
    except Exception as e:
        app.logger.error(f"An unexpected error occurred: {e}")

        return render_template("index.html", error=str(e))
    
if __name__ == '__main__':
    app.run(host="127.0.0.1", debug=True)
