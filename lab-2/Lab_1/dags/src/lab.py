import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from kneed import KneeLocator
import pickle
import os
import base64


def load_data():
    """
    Loads Formula 1 driver performance data from file.csv, serializes it, and returns base64 data.
    """
    print("Loading Formula 1 dataset...")
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/file.csv"))
    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    serialized_data = pickle.dumps(df)
    return base64.b64encode(serialized_data).decode("ascii")


def data_preprocessing(data_b64: str):
    """
    Decodes and preprocesses Formula 1 driver data for clustering.
    Uses 'points', 'position', and 'wins' as clustering features.
    """
    # Decode from base64
    data_bytes = base64.b64decode(data_b64)
    df = pickle.loads(data_bytes)

    # Drop missing or non-numeric rows
    df = df.dropna(subset=["points", "position", "wins"])

    # Select numeric columns for clustering
    clustering_data = df[["points", "position", "wins"]]

    # Scale features
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(clustering_data)

    serialized_scaled = pickle.dumps(scaled_data)
    return base64.b64encode(serialized_scaled).decode("ascii")


def build_save_model(data_b64: str, filename: str):
    """
    Builds a K-Means model for the F1 dataset and saves it to the model directory.
    Returns the SSE list for elbow method visualization.
    """
    data_bytes = base64.b64decode(data_b64)
    scaled_data = pickle.loads(data_bytes)

    sse = []
    kmeans_kwargs = {"init": "random", "n_init": 10, "max_iter": 300, "random_state": 42}

    for k in range(1, 15):  # 1–14 clusters 
        kmeans = KMeans(n_clusters=k, **kmeans_kwargs)
        kmeans.fit(scaled_data)
        sse.append(kmeans.inertia_)

    # Save the last trained model
    model_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, filename)

    with open(model_path, "wb") as f:
        pickle.dump(kmeans, f)

    return sse


def load_model_elbow(filename: str, sse: list):
    """
    Loads the saved KMeans model and determines optimal k using the elbow method.
    Returns a sample prediction from test.csv.
    """
    model_path = os.path.join(os.path.dirname(__file__), "../model", filename)
    model = pickle.load(open(model_path, "rb"))

    # Find elbow
    kl = KneeLocator(range(1, 15), sse, curve="convex", direction="decreasing")
    print(f"Optimal number of clusters (elbow): {kl.elbow}")

    # Predict for sample test data
    df_test = pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/test.csv"))
    df_test = df_test.dropna(subset=["points", "position", "wins"])
    test_scaled = MinMaxScaler().fit_transform(df_test[["points", "position", "wins"]])

    prediction = model.predict(test_scaled)[0]
    print(f"Sample prediction: Cluster {prediction}")

    return int(prediction)
