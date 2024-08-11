import torch
from torch.utils.data import Dataset


class MovieDataset(Dataset):
    def __init__(self, users, movies, ratings):
        self.users = users
        self.movies = movies
        self.ratings = ratings
        
    # len(movie_dataset)
    def __len__(self): # Number of Users
        return len(self.users)

    # movie_dataset[1]
    def __getitem__(self, idx):

        users = self.users[idx]
        movies = self.movies[idx]
        ratings = self.ratings[idx]

        return {
            "users" : torch.tensor(users, dtype=torch.long),
            "movies" : torch.tensor(movies, dtype=torch.long),
            "ratings" : torch.tensor(ratings, dtype=torch.float)
        }