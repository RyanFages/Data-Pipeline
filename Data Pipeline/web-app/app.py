from flask import Flask, request, render_template
import mlflow.pyfunc
import numpy as np

app = Flask(__name__)

mlflow.set_tracking_uri("http://mlflow-server:5000")

# Load model from MLflow
# Replace this with your actual model URI
model = mlflow.pyfunc.load_model("models:/Iris_Model/1")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sepal_width = float(request.form["sepal_width"])

    # Create input for MLflow model (expects DataFrame-like structure)
    input_data = np.array([[sepal_width]])
    
    # If your MLflow model expects a DataFrame with column names:
    import pandas as pd
    input_df = pd.DataFrame(input_data, columns=["sepal_width"])

    prediction = model.predict(input_df)[0]

    return render_template("index.html", prediction=prediction)
