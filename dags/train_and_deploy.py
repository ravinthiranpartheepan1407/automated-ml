import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline

df = pd.read_csv("/opt/airflow/datasets/tracks.csv")


def load_and_preprocess_data():
    num_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valences', 'tempos']
    X = df[num_features]
    Y = df['artist_genres'].str.get_dummies(sep=', ').idxmax(axis=1)
    return train_test_split(X, Y, test_size=0.2, random_state=42)


def train_models(X_train, Y_train):
    preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), X_train.columns.to_list())])

    models = {
        'KNN': KNeighborsClassifier(),
        'Naive Bayes': GaussianNB(),
    }

    param_grids = {
        'KNN': {
            'classifier__n_neighbors': [1, 3, 5, 7, 9, 11, 15, 20],
            'classifier__weights': ['uniform', 'distance'],
            'classifier__metric': ['euclidean', 'manhattan', 'chebyshev', 'cosine']
        },
        'Naive Bayes': {
            'classifier__var_smoothing': np.logspace(0, -9, num=10)
        },
    }

    best_model = None
    best_score = -1

    for name, model in models.items():
        with mlflow.start_run(run_name=f"{name}_GridSearchCV"):
            print(f"Running Grid Search CV for {name}...")
            pipeline = Pipeline([
                ('preprocessor', preprocessor),
                ('classifier', model)
            ])
            param_grid = param_grids[name]
            grid_search = GridSearchCV(pipeline, param_grid, cv=10, n_jobs=-1)
            grid_search.fit(X_train, Y_train)

            mlflow.log_params(grid_search.best_params_)
            mlflow.log_metric("best_cv_score", grid_search.best_score_)

            mlflow.sklearn.log_model(grid_search.best_estimator_, f"{name}_best_model")

            if grid_search.best_score_ > best_score:
                best_model = grid_search.best_estimator_
                best_score = grid_search.best_score_

    return best_model


def log_best_model(best_model):
    with mlflow.start_run(run_name="Best_Model"):
        mlflow.sklearn.log_model(best_model, "best_model")
        model_uri = mlflow.get_artifact_uri("best_model")
    return model_uri


def recommend_songs(input_genres, model, df, num_features, top_n=5):
    mask = df['artist_genres'].apply(lambda x: any(genre in x for genre in input_genres))
    filtered_data = df[mask]

    if filtered_data.empty:
        return pd.DataFrame(columns=['artist_names', 'albums', 'track_hrefs'])

    X = filtered_data[num_features]

    predictions = model.predict_proba(X)

    top_indices = np.argsort(predictions.max(axis=1))[-top_n:][::-1]
    recommendations = filtered_data.iloc[top_indices][['artist_names', 'albums', 'track_hrefs']]

    return recommendations


def main():
    mlflow.set_experiment("Music Recommendation System")

    X_train, X_test, Y_train, Y_test = load_and_preprocess_data()
    best_model = train_models(X_train, Y_train)
    model_uri = log_best_model(best_model)

    print(f"Best model URI: {model_uri}")

    # Load the model for predictions
    loaded_model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv("/opt/airflow/datasets/tracks.csv")
    num_features = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness',
                    'valences', 'tempos']
    input_genres = ['pop', 'rock']
    recommendations = recommend_songs(input_genres, loaded_model, df, num_features)
    print(recommendations)


if __name__ == "__main__":
    main()