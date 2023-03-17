from APS_Sensor.logger import logging
from APS_Sensor.Exception import SensorException
from APS_Sensor.Pipeline.TrainingPipeline import start_training_pipeline
from APS_Sensor.Pipeline.BatchPrediction import start_batch_prediction

import os
import sys

if __name__ == "__main__" :
    try:
        logging.info(f"{'==' * 15} Running Project Start...  {'==' * 15}")
        #start_training_pipeline()
        file_path = "/config/workspace/Data/aps_failure_test_set.csv"
        start_batch_prediction(input_file_path = file_path)

        logging.info(f"{'==' * 15} Running Project finished successfully! {'==' * 15}")

    except Exception as e:
       raise SensorException(e, sys)
