from setuptools import find_packages, setup

from typing import List

REQUIREMENT_FILE_NAME = 'requirements.txt'
HYPHONE_E_DOT = "-e ."

def get_requirements()-> List[str]:
    with open(REQUIREMENT_FILE_NAME) as requirement_file:
        requirement_list = requirement_file.readlines()
        requirement_list = [requirement_name.replace("\n" , "") for requirement_name in requirement_list]
        if HYPHONE_E_DOT in requirement_list :
            requirement_list.remove(HYPHONE_E_DOT)
    return requirement_list
    
setup(
name= "APS_Sensor",
version= "0.0.1",
author = "FatemehAnami",
author_email = "FatemehAnami.official@gmail.com",
packages = find_packages(),
install_requires = get_requirements(),
)
