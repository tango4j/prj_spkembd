
import numpy as np
import torch
import torchtext
from torch.utils import data
import dataloader
from dataloader import Dataset
import ipdb

from functools import partial

# CUDA for PyTorch
def getDataIndex():
    # Datasets
    partition = {'train':['MIX1008-mix04-taotA', 
                          'MIX1008-mix04-tchtA',
                          'MIX1014-mix04-tacqA'], 
                 'validation': ['MIX1300-mix04-tiubA', 
                                'MIX1300-mix04-tiuvA']}
    
    labels_path = {'MIX1008-mix04-taotA':[0,"/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/sre04/raw_mfcc_sre04.1.ark:20"],
                   'MIX1008-mix04-tchtA':[0,"/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/sre04/raw_mfcc_sre04.1.ark:620101"],
                   'MIX1014-mix04-tacqA':[2,"/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/sre04/raw_mfcc_sre04.1.ark:5580809"],
                   'MIX1300-mix04-tiubA':[3,"/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/sre04/raw_mfcc_sre04.1.ark:115954287"],
                   'MIX1300-mix04-tiuvA':[4,"/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/sre04/raw_mfcc_sre04.1.ark:116574368"]}

    return partition, labels_path

def train(config_dic):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    # cudnn.benchmark = True

    # Parameters
    params = {'batch_size': 1000,
              'shuffle': False,
              'num_workers': 6}
    max_epochs = 5
    frame_length = 1000
    fixed_frame_num = 10

    # partition, labels_path = getDataIndex()
    partition, labels_path = dataloader.dataIndexer(config_dic)
    
    # Generators
    training_set = Dataset(partition['train'], labels_path)
    training_generator = data.DataLoader(training_set, 
                                        collate_fn=partial(dataloader.my_collate, 
                                                           frame_size=frame_length, 
                                                           fixed_frame_num=fixed_frame_num), 
                                        **params)

    validation_set = Dataset(partition['validation'], labels_path)
    validation_generator = data.DataLoader(validation_set, 
                                           collate_fn=partial(dataloader.my_collate, 
                                                              frame_size=frame_length, 
                                                              fixed_frame_num=fixed_frame_num), 
                                           **params)

    # Loop over epochs
    for epoch in range(max_epochs):
	# Training
        count=0
        for local_batch, local_labels in training_generator:
            local_batch, local_labels = local_batch.to(device), local_labels.to(device)
            count += 1
            # Model computations
            print("local_batch:", local_batch.shape)
            print("local_labels:", local_labels.shape)
            print("count:", count, " epoch: ", epoch)
            input()
        # with torch.set_grad_enabled(False):
            # for local_batch, local_labels in validation_generator:
                
                # # Transfer to GPU
                # local_batch, local_labels = local_batch.to(device), local_labels.to(device)

                # # Model computations
                # print("local_batch:", local_batch.shape)
                # print("local_labels:", local_labels.shape)

