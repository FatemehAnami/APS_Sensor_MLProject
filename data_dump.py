import pymongo
import pandas as pd
import json
import os
from APS_Sensor import config

# Provide the mongodb localhost url to connect python to mongodb.
#client = pymongo.MongoClient("mongodb://localhost:27017/neurolabDB")

DATABASE_NAME = "APS_Sensor_Database"
COLLECTION_NAME = "APS_Sensor_Collection"
if __name__ == "__main__":
    df= pd.read_csv('/config/workspace/Data/aps_failure_training_set.csv')
    print('Shape of data is: ', df.shape)
        # change data to json to dump into mongodb
    df.reset_index(drop = True , inplace = True)
    json_records = list(json.loads(df.T.to_json()).values())
    print(json_records[0])

    config.mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_records)
    print("All the records added successfully")
    
    #a = client[DATABASE_NAME][COLLECTION_NAME].find()
    #print(a)
    #pd.DataFrame(list(mongo_client[database_name][collection_name].find()))

