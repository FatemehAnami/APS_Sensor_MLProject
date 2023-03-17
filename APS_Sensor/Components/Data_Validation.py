from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Entity import artifact_entity
from APS_Sensor.Exception import SensorException
from APS_Sensor.logger import logging
from APS_Sensor.config import TARGET_COLUMN
from scipy.stats import ks_2samp
from typing import Optional
import os
import sys
import pandas as pd
import numpy as np


class DataValidation:
    def __init__(self, 
                data_validation_config:config_entity.DataValidationConfig, 
                data_ingestion_artifact:artifact_entity.DataIngestionArtifact):
        try:
            logging.info(f"{'==' *20} Data Validation {'==' * 20}")
            self.data_validation_config = data_validation_config
            self.validation_report = dict()
            self.data_ingestion_artifact = data_ingestion_artifact
        except Exception as e:
            raise SensorException(e, sys)
    
    def is_required_columns_exists(self, base_df:pd.DataFrame, current_df:pd.DataFrame, report_key_name:str) -> bool:
        """
        This function check whether all columns in base dataset exists in curret dataset or not.

        base_dataframe: Pandas DataFrame
        current_dataframe: Pandas DataFrame

        return: boolean
        """
        try:
            logging.info("Find the difference between columns in base dataset and current dataset.")
            logging.info("Load the columns of base dataset and current dataset.")
            base_df_columns = base_df.columns
            current_df_columns = current_df.columns

            logging.info("Compare the columns of current dataset with base dataset.")
            missing_columns = []
            for column in base_df_columns :
                if column not in current_df_columns :
                    missing_columns.append(column)

            self.validation_report[report_key_name] = missing_columns

            logging.info(f"List of missed column are {missing_columns}.")
            if len(missing_columns) == 0 :
                return True
            else:
                return False 
        except Exception as e:
            raise SensorException(e, sys)

    def drop_missing_values_columns(self,df: pd.DataFrame,report_key_name:str)-> Optional[pd.DataFrame]:
        """
        This function drop all columns with more than threshold missing values.

        df: Pandas DataFrame
        threshhold: percentage criteria of missing value to drop a column

        return: Pandas DataFrame if a single column is available
        """
        try:
       
            threshold = self.data_validation_config.missing_threshold
            drop_column_names = []
            logging.info(f"Find list of columns that have missing value more than {threshold}")
            # Select column with more than threshold missing value
            null_report = df.isna().sum()/ df.shape[0]
            drop_column_names = list(null_report[null_report > threshold].index)
            self.validation_report[report_key_name] = drop_column_names

            logging.info(f"Drop columns {drop_column_names} with high number of missing values")
            # Drop columns with more than threshhold% null values
            df.drop(drop_column_names, axis = 1 , inplace = True)

            logging.info("Return DataFrame after dropping columns with high number of missing values")
            if len(df.columns) == 0 :
                return None
            else:
                return df        
        except Exception as e:
            raise SensorException(e, sys)

    def data_drift(self, base_df:pd.DataFrame, current_df:pd.DataFrame,report_key_name:str):
        """
        This function compare the distribution of current dataset with base dataset.

        base_dataframe: Pandas DataFrame
        current_dataframe: Pandas DataFrame

        return: Dictionary of drift report 
        """
        try:
            logging.info("Create a data dritf report")
            drift_report = dict()
            base_df_columns = base_df.columns
            current_df_columns = current_df.columns
            
            logging.info("Compare the distribution of each column in base dataset with current dataset")
            for column in base_df_columns :
                base_column_data , current_column_data = base_df[column] , current_df[column]
                logging.info(f"Test Hypothesis {column}: {base_column_data.dtype}, {current_column_data.dtype}")
                same_distribution = ks_2samp(base_column_data, current_column_data)

                # null hypothesis: both column have same distribution
                if same_distribution.pvalue > 0.05 :
                    # Accept null hypothesis
                    drift_report[column] = {
                        "Pvalue" : float(same_distribution.pvalue),
                        "SameDistribution" : True
                    }
                else:
                    # Reject null hypothesis
                    drift_report[column] = {
                        "Pvalue" : float(same_distribution.pvalue),
                        "SameDistribution" : False
                    }
               
            logging.info("Save the drift report.")
            self.validation_report[report_key_name] = drift_report
     
        except Exception as e:
            raise SensorException(e, sys)

    def initiate_data_validation(self,):
        """
        This function initiate the validation process for both train and test datasets.

        return data validation artifacts

        """
        try:
            logging.info("Load base dataset")
            base_df = pd.read_csv(self.data_validation_config.base_dataset_file_path)
            base_df.replace(to_replace = 'na', value= np.nan , inplace = True)
            logging.info("Drop high missing value columns in base dataset")
            base_df = self.drop_missing_values_columns(base_df, "Base_Dataset_dropped_columns")
            # Convert columns to float
            exclude_columns = [TARGET_COLUMN]
            utils.convert_columns_str_to_float(df=base_df, exclude_columns= exclude_columns)

            logging.info("Load Train dataset and apply validation")
            train_df = pd.read_csv(self.data_ingestion_artifact.train_file_path)
            logging.info("Drop high missing value columns in Train dataset")
            train_df = self.drop_missing_values_columns(train_df, "Train_dropped_columns")
            utils.convert_columns_str_to_float(df=train_df, exclude_columns= exclude_columns)
            train_miss_columns_status = self.is_required_columns_exists(base_df, train_df, "Train_missed_columns")
            if train_miss_columns_status:
                self.data_drift(base_df, train_df, "Train_drift_report")
            logging.info("Train dataset validation finished successfully")

            logging.info("Load Test dataset and apply validation")
            test_df = pd.read_csv(self.data_ingestion_artifact.test_file_path)
            logging.info("Drop high missing value columns in Test dataset")
            test_df = self.drop_missing_values_columns(test_df, "Test_dropped_columns")
            utils.convert_columns_str_to_float(df=test_df, exclude_columns= exclude_columns)
            test_miss_columns_status = self.is_required_columns_exists(base_df, test_df, "Test_missed_columns")
            if  test_miss_columns_status :
                self.data_drift(base_df, test_df, "Test_drift_report")
            logging.info("Test dataset validation finished successfully")

            # Write the data validation report into yaml file 
            utils.write_yaml_file(
                                file_path = self.data_validation_config.report_file_path, 
                                data = self.validation_report)
            # Prepare validation artifacts
            logging.info("Create and return validation artifacts")
            data_validation_artifact = artifact_entity.DataValidationArtifact(self.data_validation_config.report_file_path)
            logging.info(f"{'==' * 15} Data validation finished successfully! {'==' * 15}")
            return data_validation_artifact
            
        except Exception as e :
            raise SensorException(e, sys)