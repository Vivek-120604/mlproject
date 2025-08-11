import sys
from src.exception import CustomException
from src.logger import logging
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

def run_pipeline():
    try:
        logging.info("Starting the training pipeline")
        
        # Data Ingestion
        data_ingestion = DataIngestion()
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        logging.info("Data ingestion completed")
        
        # Data Transformation
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        logging.info(f"Data transformation completed. Preprocessor saved at: {preprocessor_path}")
        
        # Model Training
        model_trainer = ModelTrainer()
        model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        logging.info(f"Model training completed with score: {model_score}")
        
        return preprocessor_path
        
    except Exception as e:
        raise CustomException(e, sys)

if __name__ == "__main__":
    run_pipeline()
