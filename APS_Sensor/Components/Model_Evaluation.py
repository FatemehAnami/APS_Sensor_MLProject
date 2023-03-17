from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor.predictor import ModelResolver
from APS_Sensor.Entity import config_entity , artifact_entity
from APS_Sensor.config import TARGET_COLUMN
from APS_Sensor.utils import load_object

import os
import sys
from sklearn.metrics import f1_score
import pandas as pd
import numpy as np

class ModelEvaluation:
    def __init__(self,
                model_evaluation_config: config_entity.ModelEvaluationConfig,
                data_ingestion_artifacts: artifact_entity.DataIngestionArtifact,
                data_transformation_artifacts: artifact_entity.DataTransformationArtifact,
                model_trainer_artifacts : artifact_entity.ModelTrainerArtifact):
        '''
        This function initate the variables of class
        params:
        model_evaluation_config : load the inputs of evaluation step
        data_ingestion_artifacts: load outputs of data ingestion phase
        data_transformation_artifacts : load the outputs of data transformation phase
        model_trainer_artifact: load the outputs of model trainer phase
        '''
        try:
            logging.info(f"{'==' *20} Model Evaluation {'==' * 20}")
            self.model_evaluation_config = model_evaluation_config
            self.data_ingestion_artifacts = data_ingestion_artifacts
            self.data_transformation_artifacts = data_transformation_artifacts
            self.model_trainer_artifacts = model_trainer_artifacts
            self.model_resolver = ModelResolver()
        
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_model_evaluation(self,) -> artifact_entity.ModelEvaluationArtifact:
        '''
        This function run to initiate the Evaluation phase of project by compareing prediction
        of current model with prediction of previous model

        return evaluation artifacts
        '''
        try:
            logging.info("Check there is previous model or not")
            previous_best_saved_model = self.model_resolver.get_previous_best_saved_model_dir()
            if previous_best_saved_model is None :
                logging.info("There is not any previous model")
                model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True , improved_acuracy = None) 
                logging.info(f"Model Evaluation artifact: {model_evaluation_artifact}")
                return model_evaluation_artifact

            # loading previous model path
            logging.info("Loading previous model, transformer and target encoder path")
            transformer_path = self.model_resolver.get_previous_transformer_path()
            model_path = self.model_resolver.get_previous_model_path()
            target_encoder_path = self.model_resolver.get_previous_target_encoder_path()

            # Load previous objects
            logging.info("Load previous model, transformer and target encoder objects")
            transformer = load_object(transformer_path)
            model = load_object(model_path)
            target_encoder = load_object(target_encoder_path)

            # load Current objects
            logging.info("Load current model, transformer and target encoder objects")
            current_transformer = load_object(self.data_transformation_artifacts.transformed_object_path)
            current_model = load_object(self.model_trainer_artifacts.model_path)
            current_target_encoder = load_object(self.data_transformation_artifacts.transformed_target_encoder_path)

            # Comparison
            # predict by previous model
            logging.info("Load Test data")
            test_df = pd.read_csv(self.data_ingestion_artifacts.test_file_path)
            traget_df = test_df[TARGET_COLUMN]
            logging.info("Transform target value by target encoder object")
            y_true = target_encoder.transform(traget_df)
            logging.info("Transform features values by pipeline transformer object")
            test_array = transformer.transform(test_df.drop(TARGET_COLUMN , axis = 1))
            logging.info("Predict target value by previous model and calculate F1-score")
            y_pred = model.predict(test_array)
            previous_model_score = f1_score(y_true, y_pred)

            # predict by current model
            logging.info("Predict target value by current model and calculate F1-score")
            y_pred = current_model.predict(test_array)
            current_model_score = f1_score(y_true, y_pred)

            logging.info("Compare performance of both models")
            threshold = current_model_score - previous_model_score
            if threshold < self.model_evaluation_config.change_threshold :
                raise Exception("The current Model does not perform better than previous model")
            
            model_evaluation_artifact = artifact_entity.ModelEvaluationArtifact(is_model_accepted = True , improved_acuracy = current_model_score - previous_model_score) 
            logging.info(f"Model Evaluation artifact: {model_evaluation_artifact}")
            logging.info(f"{'==' *15} Model Evaluation finished successfully! {'==' * 15}")
            return model_evaluation_artifact

        except Exception as e:
            raise SensorException(e, sys)