import json

config_dic = {}

config_dic['config_id'] = 'salsa'
config_dic['data_type'] = 'sre'
config_dic['data_path'] = '/home/inctrl/feat_dir/nist_sre_2018/feats/mfcc20/'
config_dic['data_subfolders'] = ['sre04', 'sre05']

path_str='../configs/' + config_dic['config_id'] + '.json'
with open(path_str, 'w') as fp:
    json.dump(config_dic, fp, indent=4)
    fp.write("\n")

