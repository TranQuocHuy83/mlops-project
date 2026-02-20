import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

def train():
    X, y = load_iris(return_X_y=True)
    model = RandomForestClassifier()
    model.fit(X, y)

    mlflow.sklearn.log_model(model, "model")

if __name__ == "__main__":
    mlflow.start_run()
    train()
    mlflow.end_run()