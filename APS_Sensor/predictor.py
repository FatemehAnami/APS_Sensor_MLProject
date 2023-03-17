from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor.Entity.config_entity import TRANSFORMER_OBJECT_FILE_NAME, MODEL_FILE_NAME, TARGET_ENCODER_OBJECT_FILE_NAME
import os
import sys
from typing import Optional

class ModelResolver:

    def __init__(self, model_registery: str ="saved_models", 
                transformer_dir:str = "transformer",
                target_encoder_dir: str = "target_encoder",
                model_dir: str = 'model') :

        self.model_registery = model_registery
        self.transformer_dir = transformer_dir
        self.target_encoder_dir = target_encoder_dir
        self.model_dir = model_dir
        os.makedirs(model_registery, exist_ok = True)

    def get_previous_best_saved_model_dir(self,)-> Optional[str]:
        try:
            dir_names = os.listdir(self.model_registery)
            logging.info(f"List of previous saved model is: {dir_names}")
            if len(dir_names) == 0 :
                return None
            dir_names = list(map(int , dir_names))
            last_saved_model_path = max(dir_names)
            return os.path.join(self.model_registery , str(last_saved_model_path))
        except Exception as e:
            raise SensorException(e, sys)
    
    def get_previous_model_path(self,)-> str:
        try:
            last_saved_model_path = self.get_previous_best_saved_model_dir()
            if last_saved_model_path is None :
                raise Exception('There is not any previous saved Model')
            return os.path.join(last_saved_model_path, self.model_dir , MODEL_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_previous_transformer_path(self,)->str:
        try:
            last_saved_model_path = self.get_previous_best_saved_model_dir()
            if last_saved_model_path is None :
                raise Exception('There is not any previous Transformer')
            return os.path.join(last_saved_model_path, self.transformer_dir , TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_previous_target_encoder_path(self,)->str:
        try:
            last_saved_model_path = self.get_previous_best_saved_model_dir()
            if last_saved_model_path is None :
                raise Exception('There is not any previous target encoder')
            return os.path.join(last_saved_model_path, self.target_encoder_dir , TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_new_best_saved_model_dir(self,) -> str:
        try:
            dir_names = os.listdir(self.model_registery)
            if len(dir_names) == 0 :
                new_saved_model_path = 0
                return os.path.join(self.model_registery , str(new_saved_model_path))
            else: 
                dir_names = list(map(int , dir_names))
                new_saved_model_path = max(dir_names) + 1
                return os.path.join(self.model_registery , str(new_saved_model_path))
        except Exception as e:
            raise SensorException(e, sys)
    
    def get_new_model_path(self,)-> str:
        try:
            new_saved_model_path = self.get_new_best_saved_model_dir()
            return os.path.join(new_saved_model_path, self.model_dir , MODEL_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_new_transformer_path(self,)-> str:
        try:
            new_saved_model_path = self.get_new_best_saved_model_dir()
            return os.path.join(new_saved_model_path, self.transformer_dir , TRANSFORMER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)

    def get_new_target_encoder_path(self,) -> str :
        try:
            new_saved_model_path = self.get_new_best_saved_model_dir()
            return os.path.join(new_saved_model_path, self.target_encoder_dir , TARGET_ENCODER_OBJECT_FILE_NAME)

        except Exception as e:
            raise SensorException(e, sys)