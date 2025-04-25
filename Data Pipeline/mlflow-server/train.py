import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load and split the dataset
dataset = pd.read_csv('../data/iris.csv')
X = dataset[['sepal_width']].values
y = dataset['sepal_length'].values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Set the tracking URI to your MLflow server
mlflow.set_tracking_uri("http://localhost:5000")
experiment_name = "Iris_Experiment"
mlflow.set_experiment(experiment_name)

# Start a new MLflow run
with mlflow.start_run() as run:
    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    # Log model and metrics
    input_example = X_train[:1]
    signature = infer_signature(X_train, model.predict(X_train))

    mlflow.log_param("model_type", "Linear Regression")
    mlflow.log_param("feature", "sepal width")
    mlflow.log_metric("rmse", rmse)
    mlflow.log_metric("r2", r2)

    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name="Iris_Model",
        input_example=input_example,
        signature=signature
    )

    run_id = run.info.run_id
    print(f"Model logged. Run ID: {run_id}")

