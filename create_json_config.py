import json
import ipdb
config_dic = {}

class Machines(object):
    def __init__(self, hostname):
        self.machine_list = ['salsa', 'tango', 'hpcc']
        if hostname in self.machine_list:
            self.hostname = hostname
        else:
            self.hostname = 'default'

    def sre(self)
        config_dic['config_id'] = self.hostname
        config_dic['use_cuda'] = 1
        config_dic['CUDA_VISIBLE_DEVICES'] = "1,2,3"
        config_dic['data_type'] = 'sre_clean'
        config_dic['data_path'] = '/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20'
        config_dic['dev_set_ratio'] = 0.2
        # config_dic['model'] = 'model_baseline'
        config_dic['model'] = 'model_t'
        config_dic['model_params'] = {'input_size': 20, 
                                      'hidden_size': 128, 
                                      'num_layers': 1, 
                                      'learning_rate': 0.001}

        config_dic['max_epochs'] = 100000
        config_dic['frame_size'] = 2000
        config_dic['dataload_params'] = {'batch_size': 50,
                                         'shuffle': False,
                                         'num_workers': 8}
        # config_dic['train_subfolders'] = ['mx6', 'sre04',  'sre05',  'sre06',  'sre08',  'sre12',  'sw',  'train_aug_250k',  'vox1',  'vox2']
        config_dic['train_subfolders'] = ['sre05']
        # config_dic['dev_subfolders'] = ['sre04', 'sre05']

        return config_dic

    def default(self):
        config_dic['config_id'] = self.hostname
        config_dic['use_cuda'] = 1
        config_dic['data_type'] = 'sample_data'
        config_dic['data_path'] = './'
        config_dic['model'] = 'baseline'
        config_dic['max_epochs'] = 100000
        config_dic['dataload_params'] = {'batch_size': 10000,
                                         'shuffle': False,
                                         'num_workers': 10}
        config_dic['train_subfolders'] = ['train',  'dev']
        config_dic['data_subfolders'] = ['sample_data']
        return config_dic


def saveJson(path_str):
    with open(path_str, 'w') as fp:
        json.dump(config_dic, fp, indent=4)
        fp.write("\n")
    print("Successfully saved the json file to:", path_str)

if __name__=='__main__':
   
    ### Salsa
    hostname = 'salsa' 
    machines = Machines(hostname)
    config_id = machines.sre()

    path_str='./configs/' + config_dic['config_id'] + '.json'
    saveJson(path_str)

    ### Default
    hostname = 'default' 
    machines = Machines(hostname)
    config_id = machines.default()

    path_str='./configs/' + config_dic['config_id'] + '.json'
    saveJson(path_str)

