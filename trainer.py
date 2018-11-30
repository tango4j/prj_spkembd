
import numpy as np
import torch
import torchtext
from torch.utils import data
import dataloader
from dataloader import Dataset
import os
import ipdb
import model_baseline
from functools import partial
import models


# CUDA for PyTorch
def train(config_dic):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3" 
    device_list = [0,1,2,3]
    os.environ["CUDA_VISIBLE_DEVICES"] = config_dic['CUDA_VISIBLE_DEVICES']
    device_list = [0,1,2]
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    
    # torch.backends.cudnn.enabled=False
    # Parameters
    params = config_dic['dataload_params']
    max_epochs = config_dic['max_epochs']

    # partition, labels_path = getDataIndex()
    indexer = dataloader.dataIndexer(config_dic)
    partition, labels_path, partition_length, num_of_spks = getattr(indexer, config_dic['data_type'])()
    
    # Generator for training dataset
    training_set = Dataset(partition['train'], labels_path)
    training_generator = data.DataLoader(training_set,  **params)
    
    # Generator for test dataset
    validation_set = Dataset(partition['validation'], labels_path)
    validation_generator = data.DataLoader(validation_set, **params)

    model_file = getattr(models, config_dic['model'])
    model = model_file.init_model(config_dic, num_of_spks)

    optimizer_function = model_file.optimizer()
    optimizer = optimizer_function(model.parameters(), lr=config_dic['model_params']['learning_rate'])
    total_step_train = 1 + partition_length[0] // config_dic['dataload_params']['batch_size'] 
    total_step_valid = 1 + partition_length[1] // config_dic['dataload_params']['batch_size'] 
    
    if torch.cuda.device_count() > 1:
        print("Total ", torch.cuda.device_count(), "GPUs")
        model = torch.nn.DataParallel(model, device_ids=device_list)
    model.to(device)

    # Loop over epochs
    for epoch in range(max_epochs):
	# Training
        count=0

        for i, (X, y) in enumerate(training_generator):
            X, y = X.to(device), y.to(device)
            count += 1
            # Model computations
            print("Train local_batch:", X.shape)
            print("Train local_labels:", y.shape)
            print("count:", count, " part length:", partition_length[0], " epoch: ", epoch)
            # input()
            
            outputs = model(X, device)
            print("Outside: input size", X.size(), "output_size", outputs.size())
            loss = model_file.cal_loss(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch+1, config_dic['max_epochs'], i+1, total_step_train, loss.item()))
        with torch.set_grad_enabled(False):
            for X, y in validation_generator:
                
                # Transfer to GPU
                X, y = X.to(device), y.to(device)
                count += 1
                # Model computations
                print("Validation local_batch:", X.shape)
                print("Validation local_labels:", y.shape)
                print("count:", count, " part length:", partition_length[1], " epoch: ", epoch)

