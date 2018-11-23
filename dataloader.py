
import numpy as np
import sys
sys.path.insert(0, './kaldi-io-for-python/')
import kaldi_io
import glob
import random

import torch
from torch.utils import data
import ipdb
import modules


class Dataset(data.Dataset):
  def __init__(self, list_IDs, labels_path):
        self.list_IDs = list_IDs
        self.labels_path = labels_path

  def __len__(self):
        return len(self.list_IDs)

  def __getitem__(self, index):
        '''
        index : given list index for partition
        '''
        # Select sample
        # print("Loading data :", index)
        ID = self.list_IDs[index]
        
        # Load data and get label
        ark_path = self.labels_path[ID][1]
        X = kaldi_io.read_mat(ark_path)
        y = self.labels_path[ID][0]
        
        # X=X[:80,:].reshape((2, 40, 20))
        return X, y

def getSREspkID(new_line, dataset_name):
    '''
    There is an exception of convention for path with "sw" dataset.
    This handles the exception with "sw".

    Args:
        @new_line: actual full path of the ark file
        @dataset_name: folder name of the dataset
    Return:
        @ID: ID that identifies each utterance
        @spk_id: represents spk_id
        @path: ark file path

    '''
    ID, path = new_line.split()
    if dataset_name == 'sw':
        spk_id=''.join(ID.split('_')[:2])
    else:
        spk_id = ID.split('-')[0].split('_')[0]
    if spk_id == '':
        ipdb.set_trace()
    return ID, spk_id, path

def breakIntoFrames(mat, label, frame_size, fixed_frame_num):
    N = np.shape(mat)[0] // frame_size
    dim = np.shape(mat)[1]
    mul_mat = mat[:int(N*frame_size),:].reshape([N, frame_size, dim])

    sliced_mats, mul_label = [ mul_mat[i,:,:] for i in range(N) ], [label] * N
    if fixed_frame_num > 0:
        if N < fixed_frame_num:
            num = fixed_frame_num // N
            d = fixed_frame_num - num * N
            return sliced_mats * num + sliced_mats[:d], mul_label * num + mul_label[:d]
        if N > fixed_frame_num:
            return sliced_mats[:fixed_frame_num], mul_label[:fixed_frame_num]
        elif N == fixed_frame_num:
            return sliced_mats, mul_label
    else:
        return sliced_mats, mul_label


def my_collate(batch, frame_size, fixed_frame_num):
    '''
    Break an utterance to multiple frames.
    The last frame is dropped if len(frame) < frame_size.

    '''
    data, target = [], []
    for (mat, label) in batch:
        data_shape = np.shape(mat)
        # print("mfcc mat shape:", data_shape, "label:", label)
        if mat.shape[0] >= frame_size:
            mul_mat, mul_label = breakIntoFrames(mat, label, frame_size, fixed_frame_num)
            data.extend(mul_mat)
            target.extend(mul_label)
            # print(" mul_mat:", len(mul_mat), " mul_label:", len(mul_label))

    # print("total data len:", len(data), "total target len:", len(target))
    data = torch.LongTensor(data)
    target = torch.LongTensor(target)
    return [data, target]

def dataIndexer(config_dic):
    
    dev_set_ratio = 0.2
    print("config_Dic:", config_dic)

    partition, data_dic, labels_path, spk_set = {}, {}, {}, set()
    if config_dic["data_type"] == "sre":
        for i, dataset_name in enumerate(config_dic['train_subfolders']):
            base_path = config_dic['data_path'] + '/' + config_dic['train_subfolders'][i] + '/'
            print("dataset_name:", dataset_name)
            
            for file in glob.glob(base_path+"*.scp"):
                scpf = modules.read_txt(file)
                
                for line in scpf:
                    if len(line) > 10:
                        new_line = line.replace("/staging/sn/travadi/nist_sre_2018/feats/mfcc20", config_dic["data_path"])
                        ID, spk_id, ark_path = getSREspkID(new_line, dataset_name)
                        spk_set.add(spk_id)
                        data_dic[ID] = [spk_id, ark_path]

    spkid2index = { x:i for i, x in enumerate(spk_set) }
    for v in data_dic.keys():
        labels_path[v] = [ spkid2index[data_dic[v][0]], data_dic[v][1]]

    total_spkid = list(data_dic.keys())
    
    random.shuffle(total_spkid)
    split_point = int(len(total_spkid) * dev_set_ratio)
    
    partition['train'] = total_spkid[:split_point]
    partition['validation'] = total_spkid[split_point:]

    return partition, labels_path
    # ipdb.set_trace()
