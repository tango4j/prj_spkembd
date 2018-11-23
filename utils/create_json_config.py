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

    def sre(self):
        config_dic['config_id'] = self.hostname
        config_dic['use_cuda'] = 1
        config_dic['data_type'] = 'sre'
        config_dic['data_path'] = '/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20'
        config_dic['model'] = 'baseline'
        config_dic['train_subfolders'] = ['mx6', 'sre04',  'sre05',  'sre06',  'sre08',  'sre12',  'sw',  'train_aug_250k',  'vox1',  'vox2']
        # config_dic['dev_subfolders'] = ['sre04', 'sre05']

        return config_dic

    def default(self):
        config_dic['config_id'] = self.hostname
        config_dic['use_cuda'] = 1
        config_dic['data_type'] = 'sample_data'
        config_dic['data_path'] = './'
        config_dic['model'] = 'baseline'
        config_dic['data_subfolders'] = ['sample_data']
        return config_dic


def saveJson(path_str):
    with open(path_str, 'w') as fp:
        json.dump(config_dic, fp, indent=4)
        fp.write("\n")

if __name__=='__main__':
   
    ### Salsa
    hostname = 'salsa' 
    machines = Machines(hostname)
    config_id = machines.sre()

    path_str='../configs/' + config_dic['config_id'] + '.json'
    saveJson(path_str)

    ### Default
    hostname = 'default' 
    machines = Machines(hostname)
    config_id = machines.default()

    path_str='../configs/' + config_dic['config_id'] + '.json'
    saveJson(path_str)

