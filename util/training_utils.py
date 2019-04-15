import os
import pickle
import torch


def save_metadata(checkpoint_dir, task_parameters, net_parameters, opt_parameters, epoch_num):
    metadata_filename = os.path.join(checkpoint_dir, "experiment_metadata_{}.txt".format(epoch_num))
    with open(metadata_filename, 'w') as f:
        f.write('-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in task_parameters.items()]))

        f.write('\n-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in net_parameters.items()]))

        f.write('\n-----------------------\n')
        f.write('\n'.join(["%s = %s" % (k, v) for k, v in opt_parameters.items()]))


def save_train_log(checkpoint_dir, tab_results, epoch_num):
    logs_filename = os.path.join(checkpoint_dir, "experiment_results_{}.pkl".format(epoch_num))
    with open(logs_filename, 'wb') as f:
        pickle.dump(tab_results, f)


def get_torch_dtype():
    if torch.cuda.is_available():
        print('cuda available')
        dtypeFloat = torch.cuda.FloatTensor
        dtypeLong = torch.cuda.LongTensor
    else:
        print('cuda not available')
        dtypeFloat = torch.FloatTensor
        dtypeLong = torch.LongTensor
    return dtypeFloat, dtypeLong
