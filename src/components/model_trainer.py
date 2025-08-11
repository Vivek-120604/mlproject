import sys
import os
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
import numpy as np

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_arr, test_arr):
        try:
            logging.info("Split training and test input data")
            X_train, y_train, X_test, y_test = (
                train_arr[:, :-1],
                train_arr[:, -1],
                test_arr[:, :-1],
                test_arr[:, -1]
            )

            logging.info("Training Random Forest model")
            model = RandomForestRegressor(random_state=42)
            
            # Simple training without hyperparameter tuning for demo
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            logging.info(f"Training R2 score: {train_model_score}")
            logging.info(f"Testing R2 score: {test_model_score}")

            import os
            from src.utils import save_object
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            
            logging.info("Model training completed")
            return test_model_score
            
        except Exception as e:
            raise CustomException(e, sys)
