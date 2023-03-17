
import os 
import sys
from APS_Sensor.Exception import SensorException
from APS_Sensor.logger import logging
from datetime import datetime

STORE_FILE_NAME = "sensor.csv"
TRAIN_FILE_NAME = "train.csv"
TEST_FILE_NAME = "test.csv"
TRANSFORMER_OBJECT_FILE_NAME = 'transformer.pkl'
TARGET_ENCODER_OBJECT_FILE_NAME = 'target_encoder.pkl'
MODEL_FILE_NAME = 'model.pkl'

class TrainingPipelineConfig:
    def __init__(self):
        try:
            self.artifact_dir = os.path.join(os.getcwd(),
            'artifacts', f"{datetime.now().strftime('%m%d%Y__%H_%M_%S')}")
        except Exception  as e:
            raise SensorException(e, sys)

        
class DataIngestionConfig:
    def __init__(self, training_pipeline_config : TrainingPipelineConfig):
        try:
            self.database_name ="APS_Sensor_Database"
            self.collection_name = "APS_Sensor_Collection"
            self.data_insegtion_dir = os.path.join(training_pipeline_config.artifact_dir , "Data_Ingestion")
            self.feature_store_file_path = os.path.join(self.data_insegtion_dir, 'Feature_Store', STORE_FILE_NAME)
            self.train_file_path = os.path.join(self.data_insegtion_dir, 'Dataset', TRAIN_FILE_NAME)
            self.test_file_path = os.path.join(self.data_insegtion_dir, 'Dataset', TEST_FILE_NAME)
            self.test_size = 0.2
        except Exception as e :
            raise SensorException(e, sys) 
    
    def to_dict(self) -> dict:
        try:
            return self.__dict__
        except Exception as e:
            raise SensorException(e, sys)  


class DataValidationConfig:

    def __init__(self, training_pipeline_config : TrainingPipelineConfig):
        try:
            self.data_validation_dir = os.path.join(training_pipeline_config.artifact_dir , "Data_Validation")
            self.report_file_path = os.path.join(self.data_validation_dir, 'report.yaml')
            self.missing_threshold:float = 0.7
            self.base_dataset_file_path = os.path.join("Data/aps_failure_training_set.csv")
    
        except Exception as e :
            raise SensorException(e, sys)

class DataTransformationConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            self.data_transformation_dir = os.path.join(training_pipeline_config.artifact_dir , "Data_Transformation")
            self.transformation_object_path = os.path.join(self.data_transformation_dir, 'transformer',TRANSFORMER_OBJECT_FILE_NAME )
            self.transformation_train_path = os.path.join(self.data_transformation_dir, 'transformed', TRAIN_FILE_NAME.replace("csv", "npz"))
            self.transformation_test_path = os.path.join(self.data_transformation_dir, 'transformed', TEST_FILE_NAME.replace("csv","npz"))
            self.transformation_target_encoder_path = os.path.join(self.data_transformation_dir, 'transformed',TARGET_ENCODER_OBJECT_FILE_NAME )
    
        except Exception as e :
            raise SensorException(e, sys)

class ModelTrainerConfig:
    def __init__(self, training_pipeline_config:TrainingPipelineConfig):

        try:
            self.model_trainer_dir = os.path.join(training_pipeline_config.artifact_dir , "model_trainer")
            self.model_path = os.path.join( self.model_trainer_dir, 'model', MODEL_FILE_NAME )
            self.expected_score = 0.7
            self.overfitting_threshold = 0.1
        
        except Exception as e :
            raise SensorException(e, sys) 

class ModelEvaluationConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.change_threshold = 0.01

        except Exception as e:
            raise SensorException(e, sys) 


class ModelPusherConfig:
    def __init__(self, training_pipeline_config: TrainingPipelineConfig):
        try:
            self.model_pusher_dir = os.path.join(training_pipeline_config.artifact_dir , "model_pusher")
            self.saved_model_dir = os.path.join("saved_models")
            self.pusher_model_dir = os.path.join(self.model_pusher_dir, "saved_models")
            self.pusher_model_path = os.path.join(self.pusher_model_dir , MODEL_FILE_NAME)
            self.pusher_transformer_path = os.path.join(self.pusher_model_dir , TRANSFORMER_OBJECT_FILE_NAME)
            self.pusher_target_encoder_path = os.path.join(self.pusher_model_dir , TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)
    