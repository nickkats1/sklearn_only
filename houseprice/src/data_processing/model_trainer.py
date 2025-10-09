import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from src.config import load_config
from src.logger import logger


class ModelTrainer:
    def __init__(self, config):
        self.config = config

    def load_params(self):
        """ Loads models and hyperparameters for GridSearchCV """
        try:
            self.model_params = {
                "linear_regression_params": {
                    "copy_X": [True, False],
                    "fit_intercept": [True, False],
                    "n_jobs": [1, 5, 10, 15, None],
                    "positive": [True, False]
                },
                "lasso_regressor_params": {
                    "alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]
                },
                "ridge_regressor_params": {
                    "alpha": [1e-15, 1e-10, 1e-8, 1e-3, 1e-2, 1, 5, 10, 20, 30, 35, 40, 45, 50, 55, 100]
                },
                "random_forest_regressor_params": {
                    'n_estimators': [10, 50, 100, 200],
                    'max_depth': [5, 10, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "gradient_boosting_regressor_params": {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_samples_split': [2, 5, 10]
                },
                "bagging_regressor_params": {
                    'n_estimators': [50, 100, 200],
                    'max_samples': [1.0, 0.8, 0.6],
                    'max_features': [1.0, 0.8, 0.6]
                },
                "svm_regressor_params": {
                    "kernel": ['linear', 'poly', 'rbf'],
                    "C": [0.1, 1, 5],
                    "epsilon": [0.1, 0.2, 0.3],
                    "gamma": ['scale', 'auto']
                },
                "xgb_regressor_params": {
                    'n_estimators': [100, 200],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'max_depth': [3, 4, 5],
                    'min_child_weight': [1, 3, 5],
                    'subsample': [0.6, 0.8, 1.0],
                    'colsample_bytree': [0.6, 0.8, 1.0],
                    'gamma': [0, 0.1, 0.2]
                },
            }

            self.models = {
                "Linear Regression": LinearRegression(),
                "Ridge Regressor": Ridge(),
                "Lasso Regressor": Lasso(),
                "Random Forest Regressor": RandomForestRegressor(),
                "Gradient Boosting Regressor": GradientBoostingRegressor(),
                "Bagging Regressor": BaggingRegressor(),
                "SVM Regressor": SVR(),
                "XGB Regressor": XGBRegressor()
            }

            return self.models, self.model_params

        except Exception as e:
            logger.exception(f"Error loading models/params: {e}")
            raise

    def log_into_mlflow(self):
        """ Train models using GridSearchCV and log results to MLflow """
        try:
        
            mlflow.set_experiment("house-pipeline")

            # Load datasets
            self.df_train_scaled = pd.read_csv(self.config['processed_train'])
            self.df_test_scaled = pd.read_csv(self.config['processed_test'])
            self.y_train_df = pd.read_csv(self.config['train_target_raw'])
            self.y_test_df = pd.read_csv(self.config['test_target_raw'])

            y_train = self.y_train_df
            y_test = self.y_test_df

            for model_name, model in self.models.items():
                with mlflow.start_run(run_name=model_name):
                    # grid search params
                    param_key = model_name.lower().replace(" ", "_") + "_params"
                    params = self.model_params.get(param_key, {})

                    grid_search = GridSearchCV(model, params, cv=4, scoring="neg_mean_squared_error", n_jobs=-1)
                    grid_search.fit(self.df_train_scaled, y_train)

                    predictions = grid_search.predict(self.df_test_scaled)

                    r2 = r2_score(y_test, predictions)
                    mse = mean_squared_error(y_test, predictions)

                    # log metrics mlflow
                    mlflow.log_param("best_score", grid_search.best_score_)
                    mlflow.log_params(grid_search.best_params_)
                    mlflow.log_metric("r2_score", r2)
                    mlflow.log_metric("mse", mse)
                    mlflow.sklearn.log_model(grid_search.best_estimator_, model_name.replace(" ", "_"))

                    logger.info(f"Model: {model_name} | R2: {r2:.4f} | MSE: {mse:.4f}")

        except Exception as e:
            logger.exception(f"MLflow logging failed: {e}")
            raise


if __name__ == "__main__":
    config = load_config()
    trainer = ModelTrainer(config)
    trainer.load_params()
    trainer.log_into_mlflow()