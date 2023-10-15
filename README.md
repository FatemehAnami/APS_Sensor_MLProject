## APS Sensor Fault Detection
![image](https://github.com/FatemehAnami/APS_Sensor_MLProject/assets/97031840/66858d01-9892-4d07-821b-0db2be770389)

The APS Sensor Fault Detection is a problem of predicting Failure of Scania Trucks APS to decrease cost of reparing and also hinder dangrous events. It is modeled by various Machine learning algorithms such as Random Forest, XGBoost, Decision Tree, KNN, Logistic Regression and the best model finalized. THe hyperparameter tuning was done by Optuna library. The model deplyed by docker.  

#### Step 1 - Download dataset
```bash
To downlaod dataset run this command in terminal
wget https://archive.ics.uci.edu/ml/machine-learning-databases/00421/aps_failure_training_set.csv
```
#### Step 2 - Install the requirements
```bash
pip install -r requirements.txt
```
#### Step 3 - Run main.py file
```bash
python main.py
```
