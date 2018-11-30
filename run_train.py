'''
Project Speaker Embedding @USC

Taejin Park, taejinpa@usc.edu
Raghuveer Peri, rperi@usc.edu
Arindam Jati, jati@usc.edu

run_train.py

run_train.py executes the train.py with given parameters and settings.

'''

import numpy as np
import torch
import socket
import os
import os.path
import trainer
import json
import modules 


def getJsonConfig():
    '''
    Load a configuration file.

    '''
    config_folder = './configs/'
    hostname = socket.gethostname()
    file_directory = config_folder + hostname + '.json'
    
    if not os.path.isfile(file_directory):
        file_directory = config_folder + "default" + '.json'

    datastore = modules.loadJson(file_directory)

    return datastore

def main():
    os.system('python create_json_config.py')

    config_dic =  getJsonConfig()
    print(config_dic)
    trainer.train(config_dic)

if __name__ == '__main__':
    
    main()
