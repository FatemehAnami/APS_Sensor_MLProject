import pandas as pd
import numpy as np
from APS_Sensor.config import mongo_client
from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
import os
import sys
import yaml
import dill

def get_collection_as_dataframe(database_name:str , collection_name:str) -> pd.DataFrame :
    try:
        logging.info('Reading Data from database: {0} and collection: {1}'.format(database_name, collection_name))
        df = pd.DataFrame.from_dict(mongo_client[database_name][collection_name].find())
        logging.info(f"Finding columns: {df.columns}")
        if '_id' in df.columns :
            logging.info('Droping _id column from dataframe.')
            df = df.drop('_id', axis = 1)
            logging.info(f"Number of Rows and Columns in dataset: {df.shape}")            
        return df    
    except Exception as e:
        raise SensorException(e, sys)

def write_yaml_file(file_path, data:dict()):
        """
        This function write the validation report into a yaml file.

        file_path: str
        data: dictionary
        """
        try:
            # Create a file and save the report of data validation.
            logging.info("Create a folder for saving report into yaml file")
            file_dir = os.path.dirname(file_path)
            os.makedirs(file_dir, exist_ok = True)

            logging.info(f"Save data validation report into yaml file: {file_path}")
            with open(file_path, 'w') as file_writer :
                yaml.dump(data, file_writer)
        except Exception as e :
            raise SensorException(e, sys)

def convert_columns_str_to_float(df: pd.DataFrame , exclude_columns:list)-> pd.DataFrame:
        """
        This function convert all columns with str datatype to flot datatype
        expect the excluded ones

        df: Pandas DataFrame
        exclude_columns: list of exculded columns name

        return DataFrame with converted columns
        """
        try:
            logging.info(f"Convert str dtype columns to float dtype expect {exclude_columns}")
            for column in df.columns:
                if column not in exclude_columns:
                    df[column] = df[column].astype('float')
            logging.info("Return dataframe with converted column to float")
            return df

        except Exception as e :
            raise SensorException(e, sys)

def save_object(file_path: str, obj: object)-> None:
    """
    This function save the python object into a file. 

    file_path: file path to save object
    obj: object to save
    """
    try:
        logging.info("Enter Save object in Main.Utils")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_object:
            dill.dump(obj, file_object)
        logging.info("Exit Save object in Main.Utils")

    except Exception as e :
        raise SensorException(e, sys)

def load_object(file_path: str)-> object:
    """
    This function load object from file. 

    file_path: file path to load object
    
    return loaded object
    """
    try:
        logging.info("Enter load object in Main.Utils")
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, "rb") as file_object:
            obj=dill.load(file_object)
        logging.info("Exit load object in Main.Utils")
        return obj
        
    except Exception as e :
        raise SensorException(e, sys)

def save_numpy_array(file_path: str, array: np.array) -> None:
    """
    This function save the numpy array into a file. 

    file_path: file path to save numpy array
    array: numpy array to save
    """
    try:
        logging.info("Enter Save numpy array in Main.Utils")
        os.makedirs( os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as numpy_file:
            np.save(numpy_file, array)
        logging.info("Exit Save numpy array in Main.Utils")

    except Exception as e :
        raise SensorException(e, sys)

def load_numpy_array(file_path: str)-> np.array:
    """
    This function load numpy array from file. 

    file_path: file path to load numpy array

    return loaded numpy array 
    """    
    try:
        logging.info("Enter load numpy array in Main.Utils")
        if not os.path.exists(file_path):
            raise Exception(f"The file {file_path} does not exist")
        with open(file_path, "rb") as numpy_file:
            array=np.load(numpy_file)
        logging.info("Exit load numpy array in Main.Utils")
        return array

    except Exception as e :
        raise SensorException(e, sys)
