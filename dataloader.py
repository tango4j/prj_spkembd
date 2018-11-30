
import numpy as np
import sys
sys.path.insert(0, './kaldi-io-for-python/')
import kaldi_io
import glob
import random
import os.path 
import os
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
        ID = self.list_IDs[index]
        
        # Load data and get label
        y, ark_path, frange = self.labels_path[ID]
        X_ark = kaldi_io.read_mat(ark_path)
        X = X_ark[frange[0]:frange[1]]
        
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
    if dataset_name == 'mx6':
        spk_id=''.join(ID.split('_')[:1])
    else:
        spk_id = ID.split('-')[0].split('_')[0]
    if spk_id == '':
        ipdb.set_trace()
    return ID, spk_id, path

def breakIntoFrames(mat, label, frame_size, fixed_frame_num):
    '''
    Break one utterance into multiple sub-frames.
    If the utterance is shorter than fixed_frame_num, it drops the utterance.


    '''
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

def getFrameIndexes(ark_path, utt_len, frame_size):
    ### Nov 23, it takes too much time to use kaldi_io.read_mat.
    ### Need to replace this with other stuff

    N = utt_len//frame_size
    if N > 0 :
        indexes = [ (x*frame_size, (x+1)*frame_size) for x in range(N) ]
        return indexes, N
    else:
        return [], 0

class dataIndexer(object):
    def __init__(self, config_dic):
        self.config_dic = config_dic
   
    def sre_clean(self):
        dev_set_ratio = self.config_dic['dev_set_ratio']
        print("config_Dic:", self.config_dic)
        
        frame_size = self.config_dic['frame_size']
        partition, data_dic, labels_path, spk_set, utt_count = {}, {}, {}, set(), 0
        if self.config_dic["data_type"] == "sre_clean":
            
            # utt_len_json = 'data_index/SRE_clean_utt_len_short.json'
            utt_len_json = 'data_index/SRE_clean_utt_len.json'
            # os.system('rm ' + utt_len_json)

            # Check if there exists the utterance length index file. (It takes too much time to load it every time)
            indexExists = os.path.isfile(utt_len_json)
            if indexExists:
                utt_len_dic = modules.loadJson(utt_len_json)
            elif not indexExists:
                utt_len_dic = {}
            
            # Prepare the data for the given train_subfolders
            for i, dataset_name in enumerate(self.config_dic['train_subfolders']):
                base_path = self.config_dic['data_path'] + '/' + self.config_dic['train_subfolders'][i] + '/'
                print("Starting dataset_name:", dataset_name)
               
                # Grab all the scp files (scp: Kaldi data index file)
                for file in glob.glob(base_path+"*.scp"):
                    scpf = modules.read_txt(file)
                   
                    # Loop for each utterance (A scp file has few hundreds of utterances)
                    for line in scpf:
                        if len(line) > 2: # Make sure the line contains legit utterance information
                            
                            # Original scp files have pathnames for it's own file system. 
                            # This should be changed to your computer's setting.
                            new_line = line.replace("/staging/sn/travadi/nist_sre_2018/feats/mfcc20", self.config_dic["data_path"])
                            uttID, spk_id, ark_path = getSREspkID(new_line, dataset_name)
                            
                            # If utterance length file exists, load from there
                            if indexExists:
                                utt_len = utt_len_dic[uttID]
                            # Else, use kaldi_io to read feature length. 
                            else:
                                X = kaldi_io.read_mat(ark_path)
                                utt_len = np.shape(X)[0]
                                utt_len_dic[uttID] = utt_len

                            # Divide the feature length with the given frame_size.
                            # The features with sizes that are shorter than frame_size are dropped.
                            indexes, N = getFrameIndexes(ark_path, utt_len, frame_size)
                            
                            print("dataset name:", dataset_name, "uttID:", uttID, " utt_count:", utt_count)

                            # Loop for a single utterance. This loop creates multiple frames out of one utterance file.
                            for j, frange in enumerate(indexes):

                                # Unique ID for each frame: uttID_frame
                                uttID_frame = uttID + '-' + str(j)
                                data_dic[uttID_frame] = (spk_id, ark_path, frange)

                            utt_count += 1
                            spk_set.add(spk_id) # This is for counting the number of speakers and indexing it for train label.
                            # if len(data_dic.keys()) > 2000:
                                # break
            
            # If this is the first time scanning the data, save the utterance length file.
            if not indexExists:
                modules.saveJson(utt_len_json, utt_len_dic)
                print("Saved utt_len_json to ", utt_len_json)
        
        # Put integer label for each speaker and create labels_path for data_loader.
        spkid2index = { x:i for i, x in enumerate(spk_set) }
        for v in data_dic.keys():
            labels_path[v] = [ spkid2index[data_dic[v][0]], data_dic[v][1], data_dic[v][2] ]

        total_samples = list(data_dic.keys())
        num_of_spks = len(spk_set)
        print("Total Frames: ", len(total_samples), "Total speakers: ", num_of_spks )
        
        # Randomize to split train and validation set
        random.shuffle(total_samples)
        split_point = int(len(total_samples) * (1-dev_set_ratio))
       
        # Split the total data into two splits
        partition['train'] = total_samples[:split_point]
        partition['validation'] = total_samples[split_point:]
        # ipdb.set_trace()
        return partition, labels_path, (len(partition['train']), len(partition['validation'])), num_of_spks
