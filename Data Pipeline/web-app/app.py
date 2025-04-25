from flask import Flask, request, render_template
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import numpy as np

app = Flask(__name__)

# Load the iris dataset and train the model
iris = load_iris()
X, y = iris.data, iris.target

# We'll just use sepal width (index 1) for prediction
X_single_feature = X[:, [1]]

model = RandomForestClassifier()
model.fit(X_single_feature, y)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    sepal_width = float(request.form["sepal_width"])
    prediction = model.predict(np.array([[sepal_width]]))[0]
    return render_template("index.html", prediction=prediction)
