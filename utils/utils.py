import numpy as np
import time
from os import path as osp
import random
import torch
import os
from functools import partial
import git
import gzip
import pickle

import utils.logger as logger
import settings.consts as consts


def get_expression_list_from_file(instance_type, best=False, suffix="", eval_num=10):
    exp_list = []
    file_path = osp.join(consts.EXPRESSION_DIR, instance_type)
    if best:
        file_path += "_best"
    elif suffix:
        file_path += f"_{suffix}"
    with open(file_path, "r") as f:
        lines = f.readlines()
        for line in reversed(lines):
            exps = line.split("\t")
            exps = [x.strip() for x in exps if "inputs[" in x]
            exp_list.extend(exps)
            if len(exp_list) >= eval_num:
                break
    logger.log("\n".join(exp_list))
    return exp_list

class BranchDataset(torch.utils.data.Dataset):
    def __init__(self, root, data_num, raw_dir_name, processed_suffix):
        super().__init__()
        self.root, self.data_num = root, data_num

        self.raw_dir = osp.join(self.root, raw_dir_name)
        self.processed_dir = self.raw_dir + processed_suffix

        assert osp.exists(self.raw_dir) or osp.exists(self.processed_dir)

        if data_num > 0:
            self.load()
        else:
            self._data_list = []

    def load(self):
        os.makedirs(self.processed_dir, exist_ok=True)
        info_dict_path = osp.join(self.processed_dir, "info_dict.pt")

        if osp.exists(info_dict_path):
            info_dict = torch.load(info_dict_path)
            file_names = info_dict["file_names"]
            processed_files = info_dict["processed_files"]
        else:
            info_dict = {}
            raw_file_names = os.listdir(self.raw_dir)
            random.shuffle(raw_file_names)
            file_names = [osp.join(self.raw_dir, raw_file_name) for raw_file_name in raw_file_names]
            file_names = [x for x in file_names if not osp.isdir(x)]
            processed_files = []
            info_dict.update(processed_files=processed_files, file_names=file_names)

        if self.data_num > len(processed_files):
            for file_name in file_names[len(processed_files):self.data_num]:
                with gzip.open(file_name, 'rb') as f:
                    sample = pickle.load(f)
                processed_file = self.process_sample(sample)
                processed_files.append(processed_file)
            self._data_list = processed_files
            
            torch.save(info_dict, info_dict_path)
        else:
            self._data_list = processed_files[:self.data_num]
    
    def process_sample(self, sample):
        raise NotImplementedError

    @property
    def data(self):
        return self._data_list

    def __len__(self):
        return len(self._data_list)
    
    def __getitem__(self, idx):
        return self._data_list[idx]




def is_valid(X):
    return not (np.isnan(X).any() or np.isinf(X).any())


def normalize_features(features):
    features -= features.min(axis=0, keepdims=True)
    max_val = features.max(axis=0, keepdims=True)
    max_val[max_val == 0] = 1
    features /= max_val
    return features


def normalize_features_torch(features):
    features -= features.min(axis=0, keepdims=True)[0]
    max_val = features.max(axis=0, keepdims=True)[0]
    max_val[max_val == 0] = 1
    features /= max_val
    return features


def get_name_from_log_dir(logdir):
    splits = logdir.rstrip("/").split("/")
    exp_dir, instance_type = splits[-1], splits[-3]

    return exp_dir, instance_type


def set_seed(seed):
    assert seed >= 0
    seed = seed % 65536  # 2^16

    np.random.seed(seed)    
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    return seed


def initial_logdir_and_get_seed(exp_type, instance_type, exp_name, **kwargs):
    log_dir = get_log_dir(exp_type, instance_type, exp_name, **kwargs)
    return initial_logger_and_seed(log_dir)


def get_log_dir(exp_type, instance_type, exp_name, **kwargs):
    try:
        git_hash = git.Repo(search_parent_directories=True).head.object.hexsha[:7]
    except:
        git_hash = "no_git"

    time_str_git_hash = f"{time.strftime('%m%d%H%M%S')}_{git_hash}"
    exp_str = "_".join([f"{k}-{v}" for k,v in kwargs.items()])
    dir_name = f"{time_str_git_hash}_{exp_str}" if exp_str else time_str_git_hash

    log_dir_now = osp.join(consts.RESULT_DIR, exp_type, instance_type, exp_name, dir_name)
    return log_dir_now


def initial_logger_and_seed(log_dir, ith_exp=0, conf=None, original_seed=0):
    if original_seed < 0 or original_seed is None:
        original_seed = np.random.randint(low=0, high=4096)
    seed = original_seed + ith_exp * 1000
    seed = set_seed(seed)
    log_dir = osp.join(log_dir, f"{ith_exp}_{seed}")

    logger.configure(log_dir)

    if conf:
        conf["seed"] = seed
        logger.write_json("configs", conf)
    return log_dir, seed

def get_required_instances(instance_type, dataset_type="default", dataset_prefix="", n_instance=32, filter=None, shuffle=False):

    dataset_type = consts.TRAIN_NAME_DICT[instance_type] if dataset_type == "default" else dataset_type
    instance_dir = osp.join(consts.INSTANCE_DIR, instance_type, f"{dataset_prefix}_{dataset_type}")

    all_instance_names = os.listdir(instance_dir)
    if filter:
        assert type(filter) is str
        str_len = len(filter)
        all_instance_names = [x for x in all_instance_names if x[-str_len:] == filter]
    assert len(all_instance_names) > 0

    if len(all_instance_names) >= n_instance:
        instance_names = np.random.choice(all_instance_names, size=n_instance, replace=False) if shuffle else all_instance_names[:n_instance]
    else:
        logger.log(f"warning: exist instances {len(all_instance_names)} is less than required ({n_instance})", level=logger.WARN)
        instance_names = all_instance_names

    instance_paths = [osp.join(instance_dir, name) for name in instance_names]
    return instance_paths

def get_transition_from_expression(expression, transition_to_c=True):
    transition_func = (lambda x: f"feature[{ x.upper()}]") if transition_to_c else (lambda x: x)

    feature_set, feature_index_set = set(), set()
    begin, end = "inputs[:,", "]"
    begin_len = len(begin)
    while True:
        where_begin = expression.find(begin)
        if where_begin == -1:
            break
        where_end = expression.find(end, where_begin+begin_len)
        index = int(expression[where_begin+begin_len:where_end])
        feature_name = consts.FEATURE_NAMES[index]
        feature_index_set.add(index)
        feature_set.add(feature_name)
        expression = expression.replace(f"inputs[:,{index}]", transition_func(feature_name))

    tensor_float_str = "torch.tensor("
    tensor_float_len = len(tensor_float_str)
    while True:
        where_tensor_float_begin = expression.find(tensor_float_str)
        if where_tensor_float_begin == -1:
            break
        where_float_end = expression.find(",", where_tensor_float_begin+tensor_float_len)
        float_str = expression[(where_tensor_float_begin+tensor_float_len):where_float_end]
        where_tensor_float_end = expression.find(")", where_float_end)
        expression = expression[:where_tensor_float_begin] + float_str + expression[(where_tensor_float_end+1):]

    if transition_to_c:
        expression = expression.replace("torch.", "")
    return expression, (feature_set, feature_index_set)

get_nlp_from_expression = partial(get_transition_from_expression, transition_to_c=False)
get_c_from_expression = partial(get_transition_from_expression, transition_to_c=True)

# class MacrosGenerator():
#     def __init__(self):
#         pass