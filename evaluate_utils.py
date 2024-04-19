import numpy as np
from math import ceil
import os
from os import path as osp
import pyscipopt as scip
from time import time
from functools import wraps
import pandas as pd
import pickle

# from settings import consts
import settings.consts as consts
import utils.logger as logger
from utils.utils import normalize_features, set_seed, get_required_instances, get_c_from_expression
import utilities

from torch.multiprocessing import Pool
from multiprocessing.pool import TimeoutError
import torch


def get_preprocess_func(node_features, khali_features):
    node_features, khali_features = np.array(node_features, dtype=np.int64), np.array(khali_features, dtype=np.int64)
    def preprocess_func(node_all, khali_all):
        features = np.concatenate([node_all[:, node_features], khali_all[:, khali_features]], axis=1)
        return normalize_features(features)
    return preprocess_func

def get_score_func(score_func_str, useful_features):
    for new_i, origin_j in enumerate(useful_features):
        score_func_str = score_func_str.replace(f"inputs[:,{origin_j}]", f"new_features[:,{new_i}]")

    def score_func(new_features):
        return eval(score_func_str)

    return score_func


def load_funcs_from_str(score_func_str, simple_var=False):

    useful_features = useful_features_origin = [i for i in range(consts.TOTAL_FEATURE_NUM) if f"inputs[:,{i}]" in score_func_str]
    if simple_var:
        useful_features_name = [consts.FEATURE_NAMES[i] for i in useful_features]
        useful_features = [consts.FEATURE_NAMES_SIMPLE.index(name) for name in useful_features_name]
        useful_features_array = np.array(useful_features, dtype=np.int32)
        assert np.logical_and(useful_features_array>=0, useful_features_array < consts.TOTAL_FEATURE_NUM_SIMPLE).all()
    
    node_features = [x for x in useful_features if x < consts.NODE_FEATURE_NUM] # node feature number is always 19 whether simple var or not
    khali_features = [x-consts.NODE_FEATURE_NUM for x in useful_features if x >= consts.NODE_FEATURE_NUM]

    return get_preprocess_func(node_features, khali_features), get_score_func(score_func_str, useful_features_origin)


def get_score_func_str_from_dir(log_dir, best_i=1):
    if osp.isdir(log_dir):
        with open(osp.join(log_dir, "best.txt"), "r") as txt:
            lines = txt.readlines()
            expression = lines[-best_i].strip().split("\t")[-1]
    else:
        expression = log_dir
    return expression



class PolicyBranching(scip.Branchrule):
    NAME = "expression"
    def __init__(self, intervals=(0,0), branch_rules="relpscost"):
        super().__init__()
        self.start1, self.start2 = intervals
        self.branch_rule1, self.branch_rule2, self.branch_rule3 = [branch_rules] * 3 if type(branch_rules) is str else branch_rules

    def branchinitsol(self):
        self.ndomchgs = self.ncutoffs = 0
        self.decision_time = self.feature_time = 0.
        self.state_buffer = {}
        self.khalil_root_buffer = {}

    def branchexeclp(self, allowaddcons):
        depth = self.model.getDepth()
        if depth < self.start1:
            result = self.model.executeBranchRule(self.branch_rule1, allowaddcons)
        elif depth < self.start2:
            result = self.model.executeBranchRule(self.branch_rule2, allowaddcons)
        else:
            result = self.model.executeBranchRule(self.branch_rule3, allowaddcons)

        # fair node counting
        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

class PolicyBranchingHybridSymb(PolicyBranching):
    NAME = "hybrid-symb"
    def __init__(self, parameters, model, intervals=(0, 0), branch_rules="relpscost", device="cpu"):
        super().__init__(intervals, branch_rules)
        model.restore_state(parameters)
        model.to(device)
        model.eval()
        self.policy = model.forward
        self.device = device

    def branchexeclp(self, allowaddcons):
        depth = self.model.getDepth()
        if depth == 0:
            candidate_vars = self.model.getSymbBestCands()
            v = utilities.extract_state_fast(self.model, self.state_buffer)
            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

        if depth < self.start1:
            result = self.model.executeBranchRule(self.branch_rule1, allowaddcons)
        elif depth < self.start2:
            start_time = time()
            candidate_vars = self.model.getSymbBestCands()
            candidate_mask = [var.getCol().getIndex() for var in candidate_vars]

            v = utilities.extract_state_fast(self.model, self.state_buffer)
            v = v['values'][candidate_mask]

            state_khalil = utilities.extract_khalil_variable_features(self.model, candidate_vars, self.khalil_root_buffer)

            var_feats = np.concatenate([v, state_khalil, np.ones((v.shape[0],1))], axis=1)
            var_feats = utilities._preprocess(var_feats, mode="min-max-2")

            var_feats = torch.as_tensor(var_feats, dtype=torch.float32).to(self.device)
            self.feature_time += time() - start_time

            with torch.no_grad():
                var_logits = self.policy(var_feats).cpu().numpy()

            best_var = candidate_vars[var_logits.argmax()]
            self.model.branchVar(best_var)
            result = scip.SCIP_RESULT.BRANCHED
            self.decision_time += time() - start_time
        else:
            result = self.model.executeBranchRule(self.branch_rule3, allowaddcons)

        if result == scip.SCIP_RESULT.REDUCEDDOM:
            self.ndomchgs += 1
        elif result == scip.SCIP_RESULT.CUTOFF:
            self.ncutoffs += 1

        return {'result': result}

def evaluate_single(instance_path, brancher, seed=0, use_vanillafullstrong=False, log_detail=False, time_limit=300, gap_limit=0.0):
    m = scip.Model()
    m.readProblem(instance_path)
    utilities.init_scip_params(m, seed=seed)

    m.setIntParam('display/verblevel', 0)
    m.setIntParam('timing/clocktype', 1)  # 1: CPU user seconds, 2: wall clock time
    m.setRealParam('limits/time', time_limit)
    m.setRealParam('limits/gap', gap_limit)

    if use_vanillafullstrong:
        m.setBoolParam('branching/vanillafullstrong/integralcands', True)
        m.setBoolParam('branching/vanillafullstrong/scoreall', True)
        m.setBoolParam('branching/vanillafullstrong/collectscores', True)
        m.setBoolParam('branching/vanillafullstrong/donotbranch', True)
        m.setBoolParam('branching/vanillafullstrong/idempotent', True)

    m.includeBranchrule(
        branchrule=brancher,
        name=brancher.NAME,
        desc=f"Custom MLPOpt branching policy.",
        priority=666666, maxdepth=-1, maxbounddist=1)
    if hasattr(brancher, "so_path"):
        m.includeBranchSymbSo(brancher.so_path)

    if log_detail:
        logger.log(f"optimize begin: {instance_path}")
        m.optimize()
        logger.log(f"optimize end: {instance_path}")
    else:
        m.optimize()

    stime = m.getSolvingTime()
    nnodes = m.getNNodes()
    nlps = m.getNLPs()
    gap = m.getGap()
    status = m.getStatus()

    result = {
                'nnodes': nnodes,
                "nnodes_fair": nnodes + 2 * (brancher.ndomchgs+brancher.ncutoffs),
                'nlps': nlps,
                'stime': stime,
                'gap': gap,
                'status': consts.STATUS_DICT[status],
            }

    return result


def unpack_dict_to_kwargs_wrapper(func):
    @wraps(func)
    def new_func(kwargs_dict):
        return func(**kwargs_dict)
    return new_func


def create_so(expression, use_hash_name=True, save_dir=consts.SO_DIR, original_path=consts.SYMB_C_PATH, debug=False, compile_now=True):
    with open(original_path, "r") as original_file:
        lines = original_file.readlines()
    os.makedirs(save_dir, exist_ok=True)
    # set expression
    new_expression, (feature_set, _) = get_c_from_expression(expression)
    lines[-1] = f"{{return {new_expression};}}\n"
    # set defines
    macros = consts.get_macros(feature_set, debug=debug)
    
    hash_name = abs(hash(expression))
    name = f"exp{hash_name}" if use_hash_name else "symb_score"

    with open(osp.join(save_dir, f"{name}.c"), "w") as new_file:
        new_file.writelines(macros)
        new_file.write("\n")
        new_file.writelines(lines)

    if compile_now:
        SCIPOPTDIR = os.environ.get('SCIPOPTDIR', '').strip('"').rstrip("/")
        os.system(f"cd {save_dir};gcc -o lib{name}.so -O3 {name}.c -fPIC -shared -I{SCIPOPTDIR}/include -L{SCIPOPTDIR}/lib -lscip")

    return hash_name

INTERVALS = {}
RULES = {}
def get_brancher(agent_info, ith_processing):
    if agent_info is None:
        brancher = PolicyBranching()
        method_key = "relpscost"
    elif type(agent_info) is str:
        if "inputs" in agent_info:
            instance_type = consts.GLOBAL_INFO_DICT["instance_type"]
            intervals = INTERVALS.get(instance_type, (0, 16))

            rules = RULES.get(instance_type, ("relpscost", "symb", "relpscost"))
            eval_exp_name = consts.GLOBAL_INFO_DICT["eval_exp_name"]
            if eval_exp_name == "default":
                brancher = PolicyBranching(intervals=intervals, branch_rules=rules)
            else:
                assert eval_exp_name == "hybrid-symb"
                mlp_dict = {"parameters": osp.join(consts.ORIGINAL_MODEL_DIR, instance_type, "mlp_sigmoidal_decay", "0", "best_params.pkl"), "model": models.mlp_model.Policy()}
                brancher = PolicyBranchingHybridSymb(intervals=intervals, branch_rules=rules, **mlp_dict)
            exp_name = consts.GLOBAL_INFO_DICT['hash_name']
            method_key = f"{brancher.NAME}_{exp_name}"
            so_path = osp.join(consts.SO_DIR, f"libexp{exp_name}.so")
            assert osp.exists(so_path), agent_info
            brancher.so_path = so_path
        elif agent_info in ["pscost", "fullstrong"]:
            brancher = PolicyBranching(branch_rules=agent_info)
            method_key = agent_info
        else:
            raise NotImplementedError(f"undefined baseline name {consts.WARN_INFO_SUFFIX}")
    else:
        raise NotImplementedError(f"wrong kwargs in agent info")
    return brancher, method_key


@unpack_dict_to_kwargs_wrapper
def evaluate(instances_list, ith_processing, seed=0, agent_info=None, time_limit=300, gap_limit=0.0, use_temp_results=False, **kwargs):
    set_seed(seed)
    consts.GLOBAL_INFO_DICT.update(kwargs)

    result_list = []
    brancher, method_key = get_brancher(agent_info, ith_processing)

    failed = 0
    for instance_path in instances_list:

        splits = instance_path.split("/")
        dataset_name, dataset_type, instance_name = splits[-3], splits[-2], splits[-1].split(".")[0]
        save_dir = osp.join(consts.TEMP_RESULTS_DIR, dataset_name, method_key, f"timelimit_{time_limit}_gaplimit_{gap_limit}" if (time_limit != 300 or gap_limit != 0.0) else "")
        temp_result_path = osp.join(save_dir, f"{dataset_type}_{instance_name}.pkl")

        if use_temp_results and osp.exists(temp_result_path):
            with open(temp_result_path, 'rb') as f:
                result = pickle.load(f)
            logger.log(f"{instance_path} load previous data {consts.IMPORTANT_INFO_SUFFIX}")
        else:
            logger.log(f"instance begin: {instance_path}")
            result = evaluate_single(instance_path, brancher, seed=seed, time_limit=time_limit, gap_limit=gap_limit)

            if result["status"] != 0:
                logger.log(f"instance {instance_path} does not obtain optimal, status is: {consts.STATUS_INDEX_DICT[result['status']]} {consts.WARN_INFO_SUFFIX}", level=logger.WARN)
            result["instance_name"] = instance_path.split("/")[-1].split(".")[0]

            logger.log(f"instance end: {instance_path}")

            os.makedirs(save_dir, exist_ok=True)
            with open(temp_result_path, 'wb') as f:
                pickle.dump(result, f)

        result_list.append(result)

    assert len(result_list) + failed == len(instances_list)
    return result_list

def get_data_from_processings(pool, processings, final_time=None, fill_none=False):
    result_dict_list_list = []
    for ith_process, processing in enumerate(processings):
        try:
            result_dict_list = processing.get(None if (final_time is None) else max(final_time - time(), 0))
            result_dict_list_list.append(result_dict_list)
        except TimeoutError as e:
            logger.log(f"process {ith_process} timeout! {consts.WARN_INFO_SUFFIX}", level=logger.WARN)
            if fill_none:
                result_dict_list_list.append(None)
    pool.terminate()
    del pool, processings
    return result_dict_list_list

def get_shift_geometric_mean(data_column, shift=1):
    data_column = np.log(np.maximum(data_column+shift, 1))
    result = np.exp(data_column.mean())-shift
    return result

def get_results_statistic(result_dict_list_list, get_std=False, prefix="", name="", get_geometric_mean=True, save_csv=False):
    if not result_dict_list_list:
        logger.log("empty result_dict_list_list" + consts.WARN_INFO_SUFFIX, level=logger.WARN)
        return {}

    result_dict_list = []
    for new_result_dict_list in result_dict_list_list:
        result_dict_list.extend(new_result_dict_list)

    if not result_dict_list:
        logger.log("empty result_dict_list" + consts.WARN_INFO_SUFFIX, level=logger.WARN)
        return {}

    result_dict_list.sort(key=lambda x: x["instance_name"])
    df = pd.DataFrame(result_dict_list)
    result_dict = dict(df.mean(axis=0, numeric_only=True))

    if get_geometric_mean:
        result_dict["stime_geometric"] = get_shift_geometric_mean(df["stime"])
        result_dict["nnodes_fair_geometric"] = get_shift_geometric_mean(df["nnodes_fair"])
        result_dict["nnodes_geometric"] = get_shift_geometric_mean(df["nnodes"])

    if get_std:
        std_result_dict = dict(df.std(axis=0))
        for k,v in std_result_dict.items():
            result_dict[f"std_{k}"] = v

    if prefix:
        result_dict = {f"{prefix}/{k}":v for k,v in result_dict.items()}

    if name:
        result_dict["A_name"] = name

    if save_csv:
        file_name = f"{hash(name) if (len(name)>15) else name}.csv"
        file_name_simple = "simplified_" + file_name
        df.to_csv(osp.join(logger.get_dir(), file_name))

        df_simple = df[["instance_name", "stime", "gap", "nnodes_fair", "nnodes", "nlps"]]
        rename_dict = {k:k[2:] for k in ["Z_primaldualintegral", "Z_lp_obj", "Z_primalbound", "Z_isprimalboundsol"]}
        df_simple.rename(columns=rename_dict, inplace=True)
        df_simple.to_csv(osp.join(logger.get_dir(), file_name_simple))


    return result_dict, df


class TestAgent(object):
    def __init__(self, n_processing=1, n_processing_gpu=1, time_limit=300, gap_limit=0.0, use_temp_results=False, instance_kwargs={}, seed=0):

        self.n_processing = n_processing
        self.n_processing_gpu = n_processing_gpu
        instances_list = get_required_instances(**instance_kwargs)
        self.instance_allocation_list = [{"ith_processing": i, "use_temp_results": use_temp_results, "time_limit": time_limit, "gap_limit": gap_limit, "seed": seed, "instances_list": [instance_path], **consts.GLOBAL_INFO_DICT} for i, instance_path in enumerate(instances_list)]

    def evaluate(self, exp_name="default", agent_info=None):
        if type(agent_info) is str and "inputs" in agent_info:
            eval_exp_name = "hybrid-symb" if "hybrid-symb" in exp_name else "default"
            hash_name = create_so(agent_info)
        else:
            hash_name = eval_exp_name = None
        for instance_allocation in self.instance_allocation_list:
            instance_allocation["agent_info"] = agent_info
            instance_allocation["hash_name"] = hash_name
            instance_allocation["eval_exp_name"] = eval_exp_name

        pool = Pool(self.n_processing_gpu if "gpu" in exp_name else self.n_processing)
        processings = [pool.apply_async(evaluate, (x,)) for x in self.instance_allocation_list]
        result_dict_list_list = get_data_from_processings(pool, processings)

        result_dict, df =  get_results_statistic(result_dict_list_list, name=exp_name, save_csv=True)

        logger.log_dump_kvs(result_dict)
        return result_dict, df
