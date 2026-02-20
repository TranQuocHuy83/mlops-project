from fastapi import FastAPI
import mlflow.pyfunc

app = FastAPI()

model = mlflow.pyfunc.load_model("models:/model/Production")

@app.get("/")
def root():
    return {"message": "Model API Running"}