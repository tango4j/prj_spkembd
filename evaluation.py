import modules
import numpy as np
import scipy.spatial.distance as distance
import os
import ipdb

ebd_folder = "/home/inctrl/feat_dir/"

enr_path = "/raid/datasets/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set_V2/docs/sre18_dev_enrollment.tsv"
trials = "/raid/datasets/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set_V2/docs/sre18_dev_trials.tsv"
trials_gt = "/raid/datasets/LDC2018E46_2018_NIST_Speaker_Recognition_Evaluation_Development_Set_V2/docs/sre18_dev_trial_key.tsv"

test_ebd_model = "qodin_4gpus350spks_EP1200"
# test_ebd_model = "hecun_4gpus350spks_EP240"
# test_ebd_model = "wavap_4gpus2000spks_EP6405"
# test_ebd_model = "dewes_4gpus2500spks_EP4400"

enroll_cmn_ebd_path = ebd_folder + "sre18_enrollment/sre18_enrollment_ebd/" + test_ebd_model + "/sre18_enrollment_ebd_LN.npy"
enroll_cmn_ebd_key_path = ebd_folder + "sre18_enrollment/sre18_enrollment_ebd/" + test_ebd_model + "/sre18_enrollment_ebd_key_list.npy"
test_cmn_ebd_path = ebd_folder + "sre18_test/sre18_test_ebd/" + test_ebd_model + "/sre18_test_ebd_LN.npy"
test_cmn_ebd_key_path = ebd_folder + "sre18_test/sre18_test_ebd/" + test_ebd_model + "/sre18_test_ebd_key_list.npy"


enroll_list = modules.read_txt(enr_path)
trials_list = modules.read_txt(trials)
trials_gt_list = modules.read_txt(trials_gt)


enroll_dict_cmn_u2s = {}
enroll_dict_cmn_s2u = {}
enroll_dict_vast_u2s = {}
enroll_dict_vast_s2u = {}
type_data_format = "cmn2"


for line in enroll_list:
    spk_id = line.split('\t')[0]
    utt_file_name = line.split('\t')[1].split('.')[0]
    
    if 'sre18.sph' in line:
        enroll_dict_cmn_u2s[utt_file_name] = spk_id
        enroll_dict_cmn_s2u[spk_id] = utt_file_name
    
    elif 'sre18.flac' in line:
        enroll_dict_vast_u2s[utt_file_name] = spk_id
        enroll_dict_vast_s2u[spk_id] = utt_file_name


trial_sequence_cmn = []
trial_sequence_vast = []

for line in trials_list:
    spk_id = line.split('\t')[0]
    utt_file_name = line.split('\t')[1]
    if spk_id in enroll_dict_cmn_s2u:
        trial_sequence_cmn.append([spk_id, utt_file_name])
    elif spk_id in enroll_dict_vast_s2u:
        trial_sequence_vast.append([spk_id, utt_file_name])


trial_ground_truth_dict_cmn = {}
trial_ground_truth_dict_vast = {}

for line in trials_gt_list:
    spk_id = line.split('\t')[0]
    utt_file_name = line.split('\t')[1].split('.')[0]
    token = line.split('\t')[3]
    data_type = line.split('\t')[-1]
    if data_type == "cmn2": 
        trial_ground_truth_dict_cmn[spk_id+"@"+utt_file_name] = token
    elif data_type == "vast": 
        trial_ground_truth_dict_vast[spk_id+"@"+utt_file_name] = token

print('trial sequence cmn length: ', len(trial_sequence_cmn))
print('trial sequence ground truth dict: ', len(trial_ground_truth_dict_cmn))

enroll_cmn_ebd_mat = np.load(enroll_cmn_ebd_path)
enroll_cmn_ebd_key_list = list(np.load(enroll_cmn_ebd_key_path))

test_cmn_ebd_mat = np.load(test_cmn_ebd_path)
test_cmn_ebd_key_list = list(np.load(test_cmn_ebd_key_path))


### Make ebd dicts:
enroll_cmn_ebd_dict = {}
test_cmn_ebd_dict = {}

for i, key in enumerate(enroll_cmn_ebd_key_list):
    enroll_cmn_ebd_dict[key] = enroll_cmn_ebd_mat[i]

for i, key in enumerate(test_cmn_ebd_key_list):
    test_cmn_ebd_dict[key] = test_cmn_ebd_mat[i]

print('enroll_dict_cmn keys: ', len(enroll_dict_cmn_s2u.keys()))
print('enroll_dict_vast keys: ', len(enroll_dict_vast_s2u.keys()))

print('enroll_cmn_ebd_mat dim: ', enroll_cmn_ebd_mat.shape)
print('test_cmn_ebd_mat dim: ', test_cmn_ebd_mat.shape)


### cmn2 
cos_similarity_list = []
euc_similarity_list = []
ground_truth_list = []

eer_cal_cos_sim_list = []
eer_cal_euc_sim_list = []

eps = 1e-10

modules.cprint('Calculating the distances...', 'y')
for idx, line in enumerate(trial_sequence_cmn):
    if (idx+1) % 10000 == 0:
        print('Calculating ' + str(idx+1) + ' / ' + str(len(trial_sequence_cmn)) )
    enroll_spk_id = line[0]
    enroll_utt_file_name = enroll_dict_cmn_s2u[enroll_spk_id]
    
    test_utt_file_name = line[1].split('.')[0]
    enroll_spk_ebd = enroll_cmn_ebd_dict[enroll_dict_cmn_s2u[enroll_spk_id]]
    test_spk_ebd = test_cmn_ebd_dict[test_utt_file_name]
    
    # print("enroll_spk_ebd:", enroll_spk_ebd.shape)
    # print("test_spk_ebd: ", test_spk_ebd.shape)
    cos_sim = 1 - distance.cosine(enroll_spk_ebd, test_spk_ebd)
    euc_sim = 1/(distance.euclidean(enroll_spk_ebd, test_spk_ebd) + eps)
    
    # cos_similarity_list.append(cos_sim)
    # euc_similarity_list.append(euc_sim)
    # print( idx, '/', str(len(trial_sequence_cmn) ), 'enroll_spk_id:', enroll_spk_id, ' | test_utt_file_name: ', test_utt_file_name)
    
    gt = trial_ground_truth_dict_cmn[enroll_spk_id+"@"+test_utt_file_name]
    ground_truth_list.append(gt)
    eer_cal_cos_sim_list.append(str(cos_sim) + ' ' + gt)
    eer_cal_euc_sim_list.append(str(euc_sim) + ' ' + gt)
    # print("enroll_spk_ebd: ", enroll_spk_ebd)
    # print("test_spk_ebd: ", test_spk_ebd)

modules.write_txt('sre18_' + test_ebd_model + '_cos.eer', eer_cal_cos_sim_list)
modules.write_txt('sre18_' + test_ebd_model + '_euc.eer', eer_cal_euc_sim_list)

print('Cosine distance EER: ')
os.system('/home/inctrl/kaldi/src/ivectorbin/compute-eer sre18_' + test_ebd_model + '_cos.eer')
print('Euclidean distance EER: ')
os.system('/home/inctrl/kaldi/src/ivectorbin/compute-eer sre18_' + test_ebd_model + '_euc.eer')

