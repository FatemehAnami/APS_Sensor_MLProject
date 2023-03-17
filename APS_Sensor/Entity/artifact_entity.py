from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Entity import artifact_entity
from dataclasses import dataclass

@dataclass
class DataIngestionArtifact:
    feature_store_file_path:str
    train_file_path:str 
    test_file_path:str

@dataclass
class DataValidationArtifact:
    report_file_path:str

@dataclass
class DataTransformationArtifact:
    transformed_object_path:str
    transformed_train_path : str
    transformed_test_path: str
    transformed_target_encoder_path: str
    
@dataclass
class ModelTrainerArtifact:
    f1_train_score: float
    f1_test_score: float
    model_path: str

@dataclass    
class ModelEvaluationArtifact: 
    is_model_accepted :  bool
    improved_acuracy : float

@dataclass   
class ModelPusherArtifact:
    pusher_model_dir: str
    saved_model_dir : str


