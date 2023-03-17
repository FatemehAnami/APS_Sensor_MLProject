from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor.predictor import ModelResolver
from APS_Sensor.Entity.config_entity import ModelPusherConfig
from APS_Sensor.Entity.artifact_entity import ModelPusherArtifact, DataTransformationArtifact, ModelTrainerArtifact
from APS_Sensor.utils import load_object , save_object
import os
import sys

class ModelPusher:
    def __init__(self, 
                model_trainer_artifact: ModelTrainerArtifact,
                data_transformation_artifact : DataTransformationArtifact,
                model_pusher_config : ModelPusherConfig
                ):
        '''
        This function initiate the variables of model pusher class
        params:
        model_pusher_config: inputs of model pusher component
        data_transformation_artifact: outputs of data transformer component
        model_trainer_artifact: putputs of model trainer component

        '''
        try:
            logging.info(f"{'==' *20} Model Pusher {'==' * 20}")
            self.model_pusher_config = model_pusher_config
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_artifact = model_trainer_artifact
            self.model_resolver = ModelResolver()

        except Exception as e :
            raise SensorException(e, sys)

    def initiate_model_pusher(self):
        '''
        This function initiate all process of model pusher component

        return the artifact of model pusher component
        '''
        try:
            # load object 
            logging.info("Loading model, transformer and target encoder objects")
            transformer = load_object(self.data_transformation_artifact.transformed_object_path)
            model = load_object(self.model_trainer_artifact.model_path)
            target_encoder = load_object(self.data_transformation_artifact.transformed_target_encoder_path)

            # save objects into model pusher dir
            logging.info("Save model, transformer and target encoder objects into saved_models folder")
            save_object(self.model_pusher_config.pusher_transformer_path, transformer)
            save_object(self.model_pusher_config.pusher_model_path, model)
            save_object(self.model_pusher_config.pusher_target_encoder_path, target_encoder)

            # save objects into model resolver dir
            logging.info("Save model, transformer and target encoder objects in to artifacts folder")
            trasformer_path = self.model_resolver.get_new_transformer_path()
            model_path = self.model_resolver.get_new_model_path()
            target_encoder_path = self.model_resolver.get_new_target_encoder_path()
            save_object( trasformer_path, transformer)
            save_object(model_path , model)
            save_object( target_encoder_path , target_encoder)
            
            model_pusher_artifact = ModelPusherArtifact(pusher_model_dir=self.model_pusher_config.pusher_model_dir,
                                                        saved_model_dir = self.model_pusher_config.saved_model_dir)
        
            logging.info(f"Model Pusher Artifacts: {model_pusher_artifact}")
            logging.info(f"{'==' *15} Model Pusher finished successfully! {'==' * 15}")
            return model_pusher_artifact
            
        except Exception as e :
            raise SensorException(e, sys)