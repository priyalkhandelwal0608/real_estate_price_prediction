import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import joblib

from src.preprocessing import build_preprocessor
from src.model import get_models, evaluate_model

def train_pipeline(data_path="data/housing.csv", save_path="app/model.pkl"):
    df = pd.read_csv(data_path)

    X = df.drop("price", axis=1)
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    preprocessor = build_preprocessor()
    models = get_models()

    best_model = None
    best_score = float("inf")

    for name, model in models.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                   ('model', model)])
        pipeline.fit(X_train, y_train)
        rmse, r2 = evaluate_model(pipeline, X_test, y_test)
        print(f"{name}: RMSE={rmse:.2f}, R2={r2:.2f}")
        if rmse < best_score:
            best_score = rmse
            best_model = pipeline

    joblib.dump(best_model, save_path)
    print(f"Best model saved to {save_path}")

if __name__ == "__main__":
    train_pipeline()