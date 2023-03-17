from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Entity import artifact_entity
from APS_Sensor.Exception import SensorException
from APS_Sensor.logger import logging
from APS_Sensor.config import TARGET_COLUMN

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import LabelEncoder

from imblearn.combine import SMOTETomek

from typing import Optional
import os
import sys
import pandas as pd
import numpy as np

class DataTransformation:
    
    def __init__(self,
                 data_transformation_config: config_entity.DataTransformationConfig,
                 data_ingestion_artifact: artifact_entity.DataIngestionArtifact):
        
        try:
            logging.info(f"{'==' *20} Data Transformation {'==' * 20}")
            self.data_transformation_config = data_transformation_config
            self.data_ingestion_artifact = data_ingestion_artifact

        except Exception as e:
            raise SensorException(e, sys)

    @classmethod
    def get_data_transformation_object(cls):
        try:
            logging.info("Create pipeline for imputing missing values and scale data")
            simple_imputer = SimpleImputer(strategy = 'constant', fill_value = 0)
            robust_scaler = RobustScaler()
            pipeline = Pipeline(steps=[('Imputer', simple_imputer) , ('RobustScaler', robust_scaler)])

            return pipeline
        
        except Exception as e:
            raise SensorException(e, sys)       

    def initiate_data_transformation(self, )-> artifact_entity.DataTransformationArtifact :
            try:
                
                # Reading Training and Test 
                logging.info("Reading Train dataset")
                train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
                logging.info("Reading Test dataset")
                test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)

                #Selecting input features
                logging.info("Selecting input features of train data")
                input_feature_train_df = train_df.drop([TARGET_COLUMN], axis = 1)
                logging.info("Selecting input features of test data")
                input_feature_test_df = test_df.drop([TARGET_COLUMN], axis = 1)

                # Transforming input train and test feature
                transformation_pipleline = self.get_data_transformation_object()
                transformation_pipleline.fit(input_feature_train_df)
                logging.info("Transformig train input feature")
                input_feature_train_array = transformation_pipleline.transform(input_feature_train_df)
                logging.info("Transforming test input features ")
                input_feature_test_array = transformation_pipleline.transform(input_feature_test_df)

                # Selecting Target fearures
                logging.info("Selecting target feature of train data ")
                target_feature_train_df = train_df[TARGET_COLUMN]
                logging.info("Selectinh target feature of test data")
                target_feature_test_df = test_df[TARGET_COLUMN]

                # Encoding Target column of train and test datasets
                logging.info("Encoding Target feature of train data ")
                label_encoder = LabelEncoder()
                label_encoder.fit(target_feature_train_df)
                target_feature_train_array = label_encoder.transform(target_feature_train_df)
                logging.info("Encoding Target feature of test data ")
                target_feature_test_array = label_encoder.transform(target_feature_test_df)

                # Resampling the minority class with SMOTEomek
                logging.info(f"Size of train data before resampling: {train_df.shape} ")
                smt = SMOTETomek(random_state=42)
                # Fit the model to generate the train data.
                input_feature_train_array, target_feature_train_array = smt.fit_resample(
                                                        input_feature_train_array, target_feature_train_array)
                logging.info(f"Size of train data after resampling: {input_feature_train_array.shape} ")

                logging.info(f"Size of test data before resampling: {test_df.shape} ")
                # Fit the model to generate the test data.
                input_feature_test_array, target_feature_test_array = smt.fit_resample(
                                                        input_feature_test_array, target_feature_test_array)
                logging.info(f"Size of test data after resampling: {input_feature_test_array.shape} ")

                train_array = np.c_[input_feature_train_array , target_feature_train_array]
                test_array  = np.c_[input_feature_test_array , target_feature_test_array]

                logging.info("Saving transformed Train array")
                utils.save_numpy_array(self.data_transformation_config.transformation_train_path, train_array)
                utils.save_numpy_array(self.data_transformation_config.transformation_test_path, test_array)
                logging.info(f"Transformed Train array saved in {self.data_transformation_config.transformation_train_path} and Test array saved in {self.data_transformation_config.transformation_test_path}")

                logging.info("Saving pipeline and encoder objects")
                utils.save_object(self.data_transformation_config.transformation_object_path,transformation_pipleline)
                utils.save_object(self.data_transformation_config.transformation_target_encoder_path, label_encoder)

                # Prepare Transormation artifacts
                data_transformation_artifact = artifact_entity.DataTransformationArtifact(
                         self.data_transformation_config.transformation_object_path,
                         self.data_transformation_config.transformation_train_path,
                         self.data_transformation_config.transformation_test_path,
                         self.data_transformation_config.transformation_target_encoder_path)

                logging.info(f"{'==' * 15} Data Transfomation finished successfully! {'==' * 15}")
                return data_transformation_artifact

            except Exception as e:
                raise SensorException(e, sys)      