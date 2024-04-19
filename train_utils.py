import os
from os import path as osp
import pickle
import gzip
import torch
import numpy as np
from time import time as time
import random

import settings.consts as consts
import utils.logger as logger
import utils.utils as utils
import evaluate_utils as evaluate_utils

import torch, torch_geometric
from torch_scatter import scatter_mean, scatter_max, scatter_sum
from torch.multiprocessing import Pool
from torch_geometric.data import Data
from torch import as_tensor

from utils.rl_algos import PPOAlgo
import dso_utils.expressions as expressions_module
from dso_utils.operators import Operators
from dso_utils.symbolic_agents import DSOAgent


def get_precision(model, batch):
    X, y_label, y_index = batch.x, batch.y, batch.y_batch
    pred_y = model(X, train_mode=False)
    _, where_max = scatter_max(pred_y, y_index)
    where_illegal = (where_max == len(y_label))
    where_max[where_illegal] = 0
    real_label = y_label[where_max]
    real_label[where_illegal] = False
    return real_label

@torch.no_grad()
def get_precision_iteratively(model, data, partial_sample=None):

    scores_sum, data_sum = 0, 0
    if partial_sample is None:
        partial_sample = len(data)
    for batch in data:
        batch = batch.to(consts.DEVICE)
        batch_labels = get_precision(model, batch)
        scores_sum += batch_labels.sum(dim=-1)
        data_sum += len(batch)
        if data_sum >= partial_sample:
            break
    result = scores_sum / data_sum
    return result



class FeatureDataset(utils.BranchDataset):
    def __init__(self, root, data_num, raw_dir_name="train", processed_suffix="_feature_processed"):
        super().__init__(root, data_num, raw_dir_name, processed_suffix)

    def process_sample(self, sample):
        X, y = sample["obss"][0][0], sample["obss"][2]["scores"]
        X, useful = X[:,:-1], X[:, -1].astype(bool)
        X, y = X[useful], y[useful]
        assert utils.is_valid(X) and utils.is_valid(y)
        y = (y>=y.max())
        X = utils.normalize_features(X)
        data = Data(x=as_tensor(X, dtype=torch.float, device="cpu"), y=as_tensor(y, dtype=torch.bool, device="cpu"))
        return data



def get_all_dataset(instance_type, dataset_type=None, train_num=1000, valid_num=400, test_num=10000, batch_size_train=1000, batch_size_valid=400, batch_size_test=1000, get_train=True, get_valid=True, get_test=False):
    file_dir = osp.join(consts.SAMPLE_DIR, instance_type, consts.TRAIN_NAME_DICT[instance_type] if dataset_type is None else dataset_type)
    if get_train:
        train_dataset = FeatureDataset(file_dir, train_num)
        train_loader = torch_geometric.loader.DataLoader(train_dataset, batch_size_train, shuffle=True, follow_batch=["y"], generator=torch.Generator(device=consts.DEVICE))
    else:
        train_loader = None

    if get_valid:
        valid_dataset = FeatureDataset(file_dir, valid_num, raw_dir_name="valid")
        valid_loader = torch_geometric.loader.DataLoader(valid_dataset, batch_size_valid, shuffle=False, follow_batch=["y"])
    else:
        valid_loader = None

    if get_test:
        test_dataset = FeatureDataset(file_dir, test_num, raw_dir_name="transfer")
        test_loader = torch_geometric.loader.DataLoader(test_dataset, batch_size_test, shuffle=False, follow_batch=["y"])
    else:
        test_loader = None

    return train_loader, valid_loader, test_loader



class TrainDSOAgent(object):
    def __init__(self, 

                 seed=0,

                 batch_size=1024,
                 data_batch_size=1000,
                 eval_expression_num=48,

                 record_expression_num=16,
                 record_expression_freq=10,

                 early_stop=1000,

                 total_iter=None,
                 continue_train_path=None,

                 # env args
                 instance_kwargs={},

                 # expression
                 expression_kwargs={},

                 # agent
                 dso_agent_kwargs={},

                 # rl_algo
                 rl_algo_kwargs={},
                 ):
        self.batch_size, self.data_batch_size, self.eval_expression_num, self.seed = batch_size, data_batch_size, eval_expression_num, seed
        self.early_stop, self.current_early_stop = early_stop, 0
        self.record_expression_num, self.record_expression_freq = record_expression_num, record_expression_freq
        self.instance_type = instance_kwargs["instance_type"]

        self.total_iter = consts.ITER_DICT[self.instance_type] if total_iter is None else total_iter

        # load datasets
        self.train_data, self.valid_data, _ = get_all_dataset(**instance_kwargs)

        # expression
        self.operators = Operators(**expression_kwargs)

        # dso agent
        self.state_dict_dir, = logger.create_and_get_subdirs("state_dict")
        self.agent = DSOAgent(self.operators, **dso_agent_kwargs)
        if continue_train_path:
            logger.log(f"continue train from {continue_train_path} {consts.IMPORTANT_INFO_SUFFIX}")
            self.agent.load_state_dict(torch.load(continue_train_path))

        # rl algo
        self.rl_algo = PPOAlgo(agent=self.agent, **rl_algo_kwargs["kwargs"])

        # algo process variables
        self.train_iter = 0
        self.best_performance = - float("inf")
        self.best_writter = open(osp.join(logger.get_dir(), "best.txt"), "w")
        self.recorder = open(osp.join(logger.get_dir(), "all_expressions.txt"), "w")
        self.save_dir = osp.join(logger.get_dir(), "state_dict")

    def process(self):
        start_training_time = time()
        for self.train_iter in range(self.total_iter+1):
            if self.current_early_stop > self.early_stop:
                break

            # generate expressions
            sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list) = self.agent.sample_sequence_eval(self.batch_size)
            expression_list = [expressions_module.Expression(sequence[:length], self.operators) for sequence, length in zip(sequences, all_lengths)]

            # train
            ensemble_expressions = expressions_module.EnsemBleExpression(expression_list)
            precisions = get_precision_iteratively(ensemble_expressions, self.train_data, self.data_batch_size)

            returns, indices = torch.topk(precisions, self.eval_expression_num, sorted=False)
            sequences, all_lengths, log_probs = sequences[indices], all_lengths[indices], log_probs[indices]
            all_counters_list, all_inputs_list = [all_counters[indices] for all_counters in all_counters_list], [all_inputs[indices] for all_inputs in all_inputs_list]

            index_useful = (torch.arange(sequences.shape[1], dtype=torch.long)[None, :] < all_lengths[:, None]).type(torch.float32)
            results_rl = self.rl_algo.train(sequences, all_lengths, log_probs, index_useful, (all_counters_list, all_inputs_list), returns=returns, train_iter=self.train_iter)

            ## tensorboard record
            results = {"train/batch_best_precision": precisions.max().item(),
                       "train/batch_topk_mean_precision": precisions[indices].mean(),
                       "train/batch_topk_var_precision": precisions[indices].std(),
                       "train/batch_all_mean_precision": precisions.mean(),
                       "train/train_iteration": self.train_iter,
                       "misc/cumulative_train_time": time() - start_training_time,
                       "misc/train_time_per_iteration": (time() - start_training_time)/(self.train_iter+1)
                       }
            results.update(results_rl)

            if self.train_iter % self.record_expression_freq == 0:
                _, where_to_valid = torch.topk(precisions, self.record_expression_num, sorted=True)

                expressions_to_valid = [expression_list[i.item()] for i in where_to_valid]
                ensemble_expressions_valid = expressions_module.EnsemBleExpression(expressions_to_valid)
                precisions_valid = get_precision_iteratively(ensemble_expressions_valid, self.valid_data)
                where_to_record = torch.where(precisions_valid > self.best_performance)[0]
                if len(where_to_record) > 0:
                    self.current_early_stop = 0
                    pairs = [(expressions_to_valid[i], precisions_valid[i].item()) for i in where_to_record]
                    pairs.sort(key=lambda x: x[1])
                    self.best_performance = pairs[-1][1]
                    for (exp, value) in pairs:
                        best = f"iteration:{self.train_iter}_precision:{round(value, 4)}\t{exp.get_nlp()}\t{exp.get_expression()}\n"
                        self.best_writter.write(best)
                    logger.log(best)
                    self.best_writter.flush()
                    os.fsync(self.best_writter.fileno())
                else:
                    self.current_early_stop += self.record_expression_freq
                results.update({
                    "valid/overall_best_precision": self.best_performance, 
                    "valid/valid_best_precision": precisions_valid.max().item(),
                    "valid/valid_all_mean_precision": precisions_valid.mean(),
                    "valid/valid_all_var_precision": precisions_valid.std(),
                    "valid/valid_iteration": self.train_iter
                })


                state_dict = self.agent.state_dict()
                state_dict_save_path = osp.join(self.save_dir, f"train_iter_{self.train_iter}_precision_{round(value, 4)}.pkl")
                torch.save(state_dict, state_dict_save_path)

            logger.logkvs_tb(results)
            logger.dumpkvs_tb()
