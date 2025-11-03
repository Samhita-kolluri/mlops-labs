from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
from pathlib import Path
import pandas as pd

if __name__ == "__main__":
    # Load the Breast Cancer dataset
    data = load_breast_cancer()
    X, y = data.data, data.target
    feature_names = data.feature_names
    target_names = data.target_names

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {acc:.3f}")
    print("Classification Report:\n", classification_report(y_test, y_pred, target_names=target_names))

    # Save the model
    model_path = Path(__file__).parent / "breast_cancer_model.pkl"
    joblib.dump(model, model_path)

    # Save predictions to CSV
    df = pd.DataFrame({
        "actual": [target_names[i] for i in y_test],
        "predicted": [target_names[i] for i in y_pred]
    })
    csv_path = Path(__file__).parent / "breast_cancer_predictions.csv"
    df.to_csv(csv_path, index=False)

    print(f"Model saved at: {model_path}")
    print(f"Predictions saved at: {csv_path}")
    print("Breast Cancer model training complete!")
