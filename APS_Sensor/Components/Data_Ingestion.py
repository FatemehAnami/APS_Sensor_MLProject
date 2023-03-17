from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Entity import artifact_entity
from APS_Sensor.Exception import SensorException
from APS_Sensor.logger import logging
import os
import sys
import pandas
import numpy as np
from sklearn.model_selection import train_test_split


class DataIngestion:
    def __init__(self, data_ingestion_config: config_entity.DataIngestionConfig):
        try:
            logging.info(f"{'==' *20} Data Ingestion {'==' * 20}")
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise SensorException(e, sys)
            
    def initiate_data_ingestion(self) -> artifact_entity.DataIngestionArtifact:
        try:
            logging.info('Exporting data into pandas dataframe ')
            #Export collection data
            df: pd.DataFrame = utils.get_collection_as_dataframe(
                database_name = self.data_ingestion_config.database_name, 
                collection_name = self.data_ingestion_config.collection_name)
            
            # replace na values with null
            df.replace(to_replace = 'na', value= np.nan , inplace = True)

            logging.info("Create feature store folder if not available.")
            # Create store folder 
            feature_store_dir =os.path.dirname(self.data_ingestion_config.feature_store_file_path)
            os.makedirs(feature_store_dir , exist_ok = True)

            logging.info("Save data to feature store folder.")
            #Save fearue to feature store
            df.to_csv(path_or_buf = self.data_ingestion_config.feature_store_file_path, index = False, header = True)
            
            logging.info("Split data to train and test.")
            df_train , df_test = train_test_split(df , test_size = self.data_ingestion_config.test_size, random_state= 42)

            logging.info("Create Dataset folder to save datasets .")
            # Create store folder 
            dataset_dir = os.path.dirname(self.data_ingestion_config.train_file_path)
            os.makedirs(dataset_dir , exist_ok = True)

            logging.info("Save train and test data to dataset folder.")
            #Save fearue to feature store
            df_train.to_csv(path_or_buf = self.data_ingestion_config.train_file_path, index = False, header = True) 
            df_test.to_csv(path_or_buf = self.data_ingestion_config.test_file_path, index = False, header = True)           

            # Prepare Artifacts
            data_ingestion_artifact = artifact_entity.DataIngestionArtifact(
                feature_store_file_path=self.data_ingestion_config.feature_store_file_path,
                train_file_path=self.data_ingestion_config.train_file_path, 
                test_file_path=self.data_ingestion_config.test_file_path)
            logging.info(f"Data ingestion artifact: {data_ingestion_artifact}")
            logging.info(f"{'==' * 15} Data Ingestion finished successfully! {'==' * 15}")

            return data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)


