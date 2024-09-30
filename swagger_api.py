from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re

# FastAPI setup
app = FastAPI()

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
def create_user_genre_matrix(df, all_genres, n_users=10):
    user_genre_data = []
    for user_id in range(n_users):
        for genre in all_genres:
            tracks_with_genre = df[df['artist_genres'].apply(lambda x: genre in x)]
            if not tracks_with_genre.empty:
                interaction = np.random.choice(tracks_with_genre['track_pop'])
                user_genre_data.append([user_id, genre, interaction])

    user_genre_df = pd.DataFrame(user_genre_data, columns=['user_id', 'genre', 'interaction'])
    return user_genre_df.pivot(index='user_id', columns='genre', values='interaction').fillna(0)


# Create the user-genre interaction matrix
user_genre_matrix = create_user_genre_matrix(df, all_genres)


# Collaborative filtering function
def collaborative_filtering(user_genre_matrix, input_genres, n_similar_users=5):
    # Create a user profile based on input genres
    user_profile = pd.Series(0, index=user_genre_matrix.columns)
    for genre in input_genres:
        if genre in user_profile.index:
            user_profile[genre] = 100

    # Compute similarity between the input profile and all users
    similarity = cosine_similarity(user_profile.values.reshape(1, -1), user_genre_matrix)[0]

    # Find the most similar users
    similar_users = similarity.argsort()[::-1][:n_similar_users]

    # Get genre recommendations
    recommendations = user_genre_matrix.iloc[similar_users].mean()

    # Sort recommendations by predicted rating
    sorted_recommendations = recommendations.sort_values(ascending=False)
    return sorted_recommendations


# Function to recommend songs based on recommendations
def recommend_songs(recommendations, df, top_n=5):
    recommended_songs = []
    for genre in recommendations.index:
        songs = df[df['artist_genres'].apply(lambda x: genre in x)].sort_values('track_pop', ascending=False)
        if not songs.empty:
            recommended_songs.append(songs.iloc[0])
        if len(recommended_songs) >= top_n:
            break
    recommended_songs_df = pd.DataFrame(recommended_songs)

    # Debugging: Print the columns of the DataFrame
    print("Recommended Songs DataFrame Columns:", recommended_songs_df.columns.tolist())

    return recommended_songs_df


# Pydantic model for request body
class GenreRequest(BaseModel):
    input_genres: list[str]


# FastAPI endpoint for music recommendations
@app.post("/recommendations/")
def get_recommendations(request: GenreRequest):
    input_genres = request.input_genres
    try:
        # Check if input genres are valid
        invalid_genres = [genre for genre in input_genres if genre not in all_genres]
        if invalid_genres:
            raise HTTPException(status_code=400, detail=f"Invalid genres: {', '.join(invalid_genres)}")

        # Get recommendations
        recommendations = collaborative_filtering(user_genre_matrix, input_genres)
        recommended_songs_df = recommend_songs(recommendations, df)

        # Check and adjust column names
        if 'names' in recommended_songs_df.columns and 'artist_names' in recommended_songs_df.columns:
            return {"recommendations": recommended_songs_df[['names', 'artist_names']].values.tolist()}
        else:
            raise HTTPException(status_code=500,
                                detail="Expected columns are not found in the recommended songs DataFrame.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


# Main function for running the FastAPI app
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
