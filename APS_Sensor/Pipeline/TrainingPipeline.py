from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Components.Data_Ingestion import DataIngestion
from APS_Sensor.Components.Data_Validation import DataValidation
from APS_Sensor.Components.Data_Transformation import DataTransformation
from APS_Sensor.Components.Model_Trainer import ModelTrainer
from APS_Sensor.Components.Model_Evaluation import ModelEvaluation
from APS_Sensor.Components.Model_Pusher import ModelPusher

import os
import sys

def start_training_pipeline():
    try:
        logging.info(f"{'==' * 15} Start Training Pipeline {'==' * 15}")
        #utils.get_collection_as_dataframe('Aps_Database', 'Aps_Train')
        
        training_pipeline_config = config_entity.TrainingPipelineConfig()
        data_ingestion_config = config_entity.DataIngestionConfig(training_pipeline_config)
        #print('Ingestion Input:')
        #print(data_ingestion_config.to_dict())
        # Data Ingestion
        data_ingestion = DataIngestion(data_ingestion_config = data_ingestion_config)
        data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
        #print("Ingestion output:")
        #print(data_ingestion_artifact)
        # Data validation
        data_validation_config = config_entity.DataValidationConfig(training_pipeline_config)
        data_validtion = DataValidation(data_validation_config, data_ingestion_artifact)
        data_validation_artifact = data_validtion.initiate_data_validation()
        #print('Validation outputs')
        #print(data_validation_artifact)
        # Data Transformation
        data_transformation_config = config_entity.DataTransformationConfig(training_pipeline_config)
        data_transformation = DataTransformation(data_transformation_config, data_ingestion_artifact)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        # Model trainer
        model_trainer_config = config_entity.ModelTrainerConfig(training_pipeline_config)
        model_trainer = ModelTrainer(model_trainer_config, data_transformation_artifact)
        model_trainer_artifact = model_trainer.initiate_model_trainer()
        # Model Evaluation
        model_evaluation_config = config_entity.ModelEvaluationConfig(training_pipeline_config)
        model_evaluation = ModelEvaluation(model_evaluation_config, data_ingestion_artifact, data_transformation_artifact, model_trainer_artifact)
        model_evaluation_artifact = model_evaluation.initiate_model_evaluation()
        # Model Pusher
        model_pusher_config = config_entity.ModelPusherConfig(training_pipeline_config)
        model_pusher = ModelPusher(model_trainer_artifact, data_transformation_artifact, model_pusher_config)
        model_pusher_artifact = model_pusher.initiate_model_pusher()


        logging.info(f"{'==' * 15} The Training Pipeline finished successfully! {'==' * 15}")

    except Exception as e:
        raise SensorException(e, sys)
