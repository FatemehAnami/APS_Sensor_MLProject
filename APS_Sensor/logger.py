import logging
import os
from datetime import datetime

# Create a format for name of log files
LOG_FILE_NAME = f"{datetime.now().strftime('%m%d%Y__%H_%M_%S')}.log"

# Path of the directory for saving logs
LOG_FILE_DIR  = os.path.join(os.getcwd(),"logs")

LOG_FILE_PATH = os.path.join(LOG_FILE_DIR , LOG_FILE_NAME)

# Create folder for saving Log Files
os.makedirs(LOG_FILE_DIR, exist_ok=True)

# Set logging 
logging.basicConfig(
    filename= LOG_FILE_PATH,
    format= "[ %(asctime)s ] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level = logging.INFO
)

