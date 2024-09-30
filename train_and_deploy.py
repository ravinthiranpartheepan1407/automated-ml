import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_squared_error
import re
import dill as pickle  # Use dill for better serialization

# Set up MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")  # Replace with your MLflow tracking server URL if different
mlflow.set_experiment("Music Recommendation System")

# Load and preprocess data
df = pd.read_csv("./data/tracks.csv")

# Function to parse the genre string into a list
def parse_genres(genre_string):
    return re.findall(r"'(.*?)'", genre_string)

# Convert string representation of list to actual list
df['artist_genres'] = df['artist_genres'].apply(parse_genres)

# Flatten the genre lists and get unique genres
all_genres = set([genre for genres in df['artist_genres'] for genre in genres])

# Create user-genre interaction matrix
def create_user_genre_matrix(df, all_genres, n_users=100):
    user_genre_data = []
    for user_id in range(n_users):
        for genre in all_genres:
            tracks_with_genre = df[df['artist_genres'].apply(lambda x: genre in x)]
            if not tracks_with_genre.empty:
                interaction = np.random.choice(tracks_with_genre['track_pop'])
                user_genre_data.append([user_id, genre, interaction])

    user_genre_df = pd.DataFrame(user_genre_data, columns=['user_id', 'genre', 'interaction'])
    return user_genre_df.pivot(index='user_id', columns='genre', values='interaction').fillna(0)

# Collaborative Filtering model
class CollaborativeFilteringModel:
    def __init__(self, user_genre_matrix, n_similar_users=5):
        self.user_genre_matrix = user_genre_matrix
        self.n_similar_users = n_similar_users

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        recommendations = []
        for _, user_profile in X.iterrows():
            similarity = cosine_similarity(user_profile.values.reshape(1, -1), self.user_genre_matrix)[0]
            similar_users = similarity.argsort()[::-1][:self.n_similar_users]
            user_recommendations = self.user_genre_matrix.iloc[similar_users].mean()
            recommendations.append(user_recommendations)
        return np.array(recommendations)

# Main execution
if __name__ == "__main__":
    with mlflow.start_run():
        # Create user-genre matrix
        user_genre_matrix = create_user_genre_matrix(df, all_genres)

        # Split the data
        train_matrix, test_matrix = train_test_split(user_genre_matrix, test_size=0.2, random_state=42)

        # Create and train the model
        model = CollaborativeFilteringModel(train_matrix)
        model.fit(train_matrix)

        # Make predictions
        predictions = model.predict(test_matrix)

        # Calculate MSE
        mse = mean_squared_error(test_matrix.values, predictions)
        print(f"Mean Squared Error: {mse}")

        # Log metrics
        mlflow.log_metric("mean_squared_error", mse)

        # Log parameters
        mlflow.log_param("n_similar_users", model.n_similar_users)

        # Save the model
        with open("models/collaborative_filtering_model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Log the model file to MLflow
        mlflow.log_artifact("models/collaborative_filtering_model.pkl")
