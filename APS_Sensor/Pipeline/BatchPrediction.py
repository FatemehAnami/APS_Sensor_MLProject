from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor.predictor import ModelResolver
from APS_Sensor.utils import load_object, save_numpy_array
from APS_Sensor.config import TARGET_COLUMN
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

PREDICTION_DIR = "prediction"

def start_batch_prediction(input_file_path: str)-> str:
    try:
        logging.info(f"{'==' *20} Prediction {'==' * 20}")
        logging.info("Create Prediction dir for saving predictioons")
        os.makedirs(PREDICTION_DIR , exist_ok = True)
        logging.info("Creating model resolver object")
        model_resolver = ModelResolver(model_registery='saved_models')
        logging.info(f"Reading data file for prediction: {input_file_path}")
        df = pd.read_csv(input_file_path)
        df.replace('na', np.nan, inplace = True)

        logging.info("Loading transformer to transform dataset")
        transformer = load_object(model_resolver.get_previous_transformer_path())
        input_array = transformer.transform(df.drop(TARGET_COLUMN, axis = 1))


        logging.info("Loading model to make prediction")
        model = load_object(model_resolver.get_previous_model_path())
        y_pred = model.predict(input_array)

        logging.info("Loading target encoder to encode predicted value to categorical")
        target_encoder = load_object(model_resolver.get_previous_target_encoder_path())
        cat_y_pred = target_encoder.inverse_transform(y_pred)

        logging.info("Add predicted values to dataframe")
        df["prediction"] = y_pred
        df["cat_prediction"] = cat_y_pred

        logging.info("Save dataframe with prediction valuse")
        prediction_file_name = os.path.basename(input_file_path).replace('.csv', f"{datetime.now().strftime('%m%d%Y__%H%M%S')}.csv")
        prediction_file_path = os.path.join(PREDICTION_DIR , prediction_file_name)
        df.to_csv(prediction_file_path, index = False , header = True)
        logging.info(f"{'==' *15} Prediction finished successfully! {'==' * 15}")
        return prediction_file_path
    
    except Exception as e:
        raise SensorException(e, sys)