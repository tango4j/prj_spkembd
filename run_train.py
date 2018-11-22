'''
Project Speaker Embedding.

Taejin Park, taejinpa@usc.edu
Raghuveer Peri, rperi@usc.edu
Arindam Jati, jati@usc.edu

run_train.py

it executes the train.py with given parameters and settings.

'''

import numpy as np
import torch
# import torchtext
import socket
import os.path
import kaldi_io

def getJsonConfig():
    config_folder = './configs/'
    hostname = socket.gethostname()
    file_directory = config_folder + hostname + '.json'
    if not os.path.isfile(file_directory):
        file_directory = config_folder + "default" + '.json'
    return open(file_directory).read()

def main():
    config_json_dic =  getJsonConfig()
    print(config_json_dic)

if __name__ == '__main__':
    main()
