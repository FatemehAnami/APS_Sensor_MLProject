from APS_Sensor import utils
from APS_Sensor.Entity import config_entity
from APS_Sensor.Entity import artifact_entity
from APS_Sensor.Exception import SensorException
from APS_Sensor.logger import logging

from typing import Optional
import os
import sys
import pandas as pd
import numpy as np

from xgboost import XGBClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import optuna


class ModelTrainer:
    def __init__(self, 
                model_trainer_config: config_entity.ModelTrainerConfig,
                data_transformation_artifact : artifact_entity.DataTransformationArtifact):
        try:
            logging.info(f"{'==' *20} Model Trainer {'==' * 20}")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

        except Exception as e:
            raise SensorException(e, sys)
    
    def hyperparameters_tuning(self, trial):
        try:
            logging.info("Hyperparameters tuning of model with optuna")
            param  = {
                'tree_method' :'hist',
                'verbosity' : 3,
                'objective' : "binary:logistic",
                'booster' : trial.suggest_categorical('booster' , ['dart' , 'gbtree','gblinear']),
                'lambda' : trial.suggest_float('lambda' , 1e-4 , 1),
                'alpha' :trial.suggest_float('alpha' , 1e-4 , 1),
                'subsample' : trial.suggest_float('subsample' , .1,.5),
                'colsample_bytree' : trial.suggest_float('colsample_bytree' , .1 ,.5)
            }

            if param['booster'] in ['gbtree' , 'dart']:
                param['gamma'] :trial.suggest_float('gamma' , 1e-3 , 4 )
                param['eta'] : trial.suggest_float('eta' , .001 ,5 )

            logging.info("Load data for hyperparameter tuning")
            data = utils.load_numpy_array(self.data_transformation_artifact.transformed_train_path)
            x , y =  data[:,:-1], data[: , -1]

            logging.info("Split data into train and validation checking accuracy of model based on selected parameters")
            train_x , test_x , train_y , test_y= train_test_split(x , y , test_size = .20 ,random_state=1)
            logging.info("Train the model with different set of parameters")
            xgboost_clf = XGBClassifier(**param)
            xgboost_clf.fit(train_x , train_y)
            accuracy  = xgboost_clf.score(test_x , test_y)
            logging.info(f"Accuracy of model on set of parameters is: {accuracy}")

            return accuracy

        except Exception as e:
            raise SensorException(e, sys)

    def train_model(self, x , y, params=None):
        try:
            if params == None :
                logging.info("Create XGBoost Classifier base model")
                xgboost_clf = XGBClassifier()
                logging.info("Fit XGBoost Classifier base model")
                xgboost_clf.fit(x , y)
                return xgboost_clf
            else :
                logging.info("Create XGBoost Classifier model with tunned parameters")
                xgboost_clf = XGBClassifier(**params)
                logging.info("Fit XGBoost Classifier model with tunned parameters")
                xgboost_clf.fit(x , y)
                return xgboost_clf
 
        except Exception as e:
            raise SensorException(e, sys)


    def initiate_model_trainer(self)-> artifact_entity.ModelTrainerArtifact:
        try:
            logging.info("Load Train and Test data ")
            train_data = utils.load_numpy_array(self.data_transformation_artifact.transformed_train_path)
            test_data = utils.load_numpy_array(self.data_transformation_artifact.transformed_test_path)

            logging.info("Create X_train, y_train and x_test, y_test data")
            x_train, y_train = train_data[:,:-1], train_data[: , -1]
            x_test , y_test  = test_data[:, :-1] , test_data[: , -1]

            # Train the base model 
            base_model = self.train_model(x_train, y_train)
            # Predict the model on train data
            logging.info("Predict target feature of train data with base model")
            base_y_hat_train = base_model.predict(x_train)
            base_f1_train_score = f1_score(y_train, base_y_hat_train)
            logging.info(f"F1-score of base model on train data is: {base_f1_train_score}")

            # Evaluate the base model on test data
            logging.info("Predict target feature of test data with base model")
            base_y_hat_test  = base_model.predict(x_test)
            base_f1_test_score = f1_score(y_test, base_y_hat_test)
            logging.info(f"F1-score of base model on test data is: {base_f1_test_score}")

            # Hyperparameters tuning
            #oputuna_model = optuna.create_study()
            #oputuna_model.optimize(self.hyperparameters_tuning, n_trials = 50 )
            #params = oputuna_model.best_trial.params
            #logging.info(f"The best parametr for model are: {params}")

            # Train the 
            #logging.info("Train the model with best parameters")
            #model = self.train_model(x_train, y_train, params)
            # Predict the model on train data
            #logging.info("Predict target feature of trian data")
            #y_hat_train = model.predict(x_train)
            #f1_train_score = f1_score(y_train, y_hat_train)
            #logging.info(f"F1-score of model on train data is: {f1_train_score}")

            # Evaluate the model on test data
            #logging.info("Predict target feature of test data")
            #y_hat_test  = model.predict(x_test)
            #f1_test_score = f1_score(y_test, y_hat_test)
            #logging.info(f"F1-score of model on test data is: {f1_test_score}")

            #flag = 1
            #if base_f1_test_score > f1_test_score: 
            #    logging.info("Base model perform better than tunned model")
            flag = 0
            # Check for overfitting or under fitting
            if flag == 1 :
                if f1_test_score < self.model_trainer_config.expected_score:
                    raise Exception(f"Model is not good and its actual score is {f1_test_score} \
                        while expected score is {self.model_trainer_config.expected_score}")
                diff_score = abs(f1_train_score - f1_test_score)
                if  diff_score > self.model_trainer_config.overfitting_threshold:
                    raise Exception(f"The model score difference between train and test data \
                        is {diff_score} which is more than overfitting threshold")
                logging.info("Save the best created model")            
                utils.save_object(self.model_trainer_config.model_path, model)
                # Create model traner artifact
                model_trainer_artifact = artifact_entity.ModelTrainerArtifact(f1_train_score,
                                                                f1_test_score, 
                                                                self.model_trainer_config.model_path)
            else: 
                if base_f1_test_score < self.model_trainer_config.expected_score:
                    raise Exception(f"Model is not good and its actual score is {base_f1_test_score} \
                        while expected score is {self.model_trainer_config.expected_score}")
                diff_score = abs(base_f1_train_score - base_f1_test_score)
                if  diff_score > self.model_trainer_config.overfitting_threshold:
                    raise Exception(f"The model score difference between train and test data \
                        is {diff_score} which is more than overfitting threshold")

                logging.info("Save the created model")            
                utils.save_object(self.model_trainer_config.model_path, base_model)

                # Create model traner artifact
                model_trainer_artifact = artifact_entity.ModelTrainerArtifact(base_f1_train_score,
                                                                base_f1_test_score, 
                                                                self.model_trainer_config.model_path)
        
            logging.info(f"{'==' * 15} Model Trainer finished successfully! {'==' * 15}")
            return model_trainer_artifact

        except Exception as e:
            raise SensorException(e, sys)