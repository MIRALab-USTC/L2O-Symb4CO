from typing import Any
import torch
import os
from os import path as osp
from collections import defaultdict
import hydra
# donot import numpy here as we have to set np threads later

IMPORTANT_INFO_SUFFIX = "*"*10
WARN_INFO_SUFFIX = "!"*10

NODE_FEATURE_NUM = 19
KHALI_FEATURE_NUM = 72
WHERE_PSEUDO = NODE_FEATURE_NUM + 52
TOTAL_FEATURE_NUM = NODE_FEATURE_NUM + KHALI_FEATURE_NUM

NODE_FEATURES = ['type_0', 'type_1', 'type_2', 'type_3', 'coef_normalized', 'has_lb', 'has_ub', 'sol_is_at_lb', 'sol_is_at_ub', 'sol_frac', 'basis_status_0', 'basis_status_1', 'basis_status_2', 'basis_status_3', 'reduced_cost', 'age', 'sol_val', 'inc_val', 'avg_inc_val']
KHALI_FEATURES = ['acons_max1', 'acons_max2', 'acons_max3', 'acons_max4', 'acons_mean1', 'acons_mean2', 'acons_mean3', 'acons_mean4', 'acons_min1', 'acons_min2', 'acons_min3', 'acons_min4', 'acons_nb1', 'acons_nb2', 'acons_nb3', 'acons_nb4', 'acons_sum1', 'acons_sum2', 'acons_sum3', 'acons_sum4', 'acons_var1', 'acons_var2', 'acons_var3', 'acons_var4', 'cdeg_max', 'cdeg_max_ratio', 'cdeg_mean', 'cdeg_mean_ratio', 'cdeg_min', 'cdeg_min_ratio', 'cdeg_var', 'coefs', 'coefs_neg', 'coefs_pos', 'frac_down_infeas', 'frac_up_infeas', 'nb_down_infeas', 'nb_up_infeas', 'nnzrs', 'nrhs_ratio_max', 'nrhs_ratio_min', 'ota_nn_max', 'ota_nn_min', 'ota_np_max', 'ota_np_min', 'ota_pn_max', 'ota_pn_min', 'ota_pp_max', 'ota_pp_min', 'prhs_ratio_max', 'prhs_ratio_min', 'ps_down', 'ps_product', 'ps_ratio', 'ps_sum', 'ps_up', 'root_cdeg_max', 'root_cdeg_mean', 'root_cdeg_min', 'root_cdeg_var', 'root_ncoefs_count', 'root_ncoefs_max', 'root_ncoefs_mean', 'root_ncoefs_min', 'root_ncoefs_var', 'root_pcoefs_count', 'root_pcoefs_max', 'root_pcoefs_mean', 'root_pcoefs_min', 'root_pcoefs_var', 'slack', 'solfracs']

CONSTRAINT_FEATURES = ['obj_cosine_similarity', 'bias', 'is_tight', 'age', 'dualsol_val_normalized']
EDGE_FEATURES = ['coef_normalized']


assert len(KHALI_FEATURES) == KHALI_FEATURE_NUM and len(NODE_FEATURES) == NODE_FEATURE_NUM
FEATURE_NAMES = NODE_FEATURES + KHALI_FEATURES

TRAIN_NAME_DICT = defaultdict(lambda : "")
TRAIN_NAME_DICT.update({"setcover": "500r_1000c_0.05d", "indset": "750_4", "facilities": "100_100_5", "cauctions": "100_500"})

WORK_DIR = osp.dirname(osp.dirname(__file__))
DATA_DIR = osp.join(WORK_DIR, "data")
RESULT_BASE_DIR = osp.join(WORK_DIR, "results")

INSTANCE_DIR = osp.join(DATA_DIR, "instances")
SAMPLE_DIR = osp.join(DATA_DIR, "samples")

RESULT_DIR = osp.join(RESULT_BASE_DIR, "results")
EXPRESSION_DIR = osp.join(RESULT_BASE_DIR, "expressions")

CONTINUE_TRAIN_SAVE_DIR = osp.join(RESULT_BASE_DIR, "state_dicts")
TEMP_RESULTS_DIR = osp.join(RESULT_BASE_DIR, "temp_results")
SO_DIR = osp.join(RESULT_BASE_DIR, "libs")
SYMB_C_PATH = osp.join(WORK_DIR, "utils", "symb_score_original.c")
os.makedirs(EXPRESSION_DIR, exist_ok=True)


ITER_DICT = defaultdict(lambda : 3000)
START_EVAL_ITER_DICT = defaultdict(lambda : 10000)
GLOBAL_INFO_DICT = {}


SAFE_EPSILON = 1e-6
DETAILED_LOG_FREQ = 100
DETAILED_LOG = True

STATUS_DICT = {"optimal": 0, "timelimit":-1, "infeasible":-2, "unbounded":-3, "userinterrupt":-4, "unknown":-5}
STATUS_INDEX_DICT = {v:k for k,v in STATUS_DICT.items()}

class Macros:
    USE_ROOT_CDEG_MEAN = set(['ROOT_CDEG_MEAN', 'ROOT_CDEG_VAR', 'CDEG_MEAN_RATIO'])
    USE_ROOT_CDEG_MIN = set(['ROOT_CDEG_MIN', 'CDEG_MIN_RATIO'])
    USE_ROOT_CDEG_MAX = set(['ROOT_CDEG_MAX', 'CDEG_MAX_RATIO'])
    USE_ROOT = set(['TYPE_0', 'TYPE_1', 'TYPE_2', 'TYPE_3', 'COEFS', 'COEFS_NEG', 'COEFS_POS', 'NNZRS', 'ROOT_CDEG_MAX', 'ROOT_CDEG_MEAN', 'ROOT_CDEG_MIN', 'ROOT_CDEG_VAR', 'ROOT_NCOEFS_COUNT', 'ROOT_NCOEFS_MAX', 'ROOT_NCOEFS_MEAN', 'ROOT_NCOEFS_MIN', 'ROOT_NCOEFS_VAR', 'ROOT_PCOEFS_COUNT', 'ROOT_PCOEFS_MAX', 'ROOT_PCOEFS_MEAN', 'ROOT_PCOEFS_MIN', 'ROOT_PCOEFS_VAR'])
    USE_TYPE = set(['TYPE_0', 'TYPE_1', 'TYPE_2', 'TYPE_3'])

    USE_ROOT_CDEG = set(['ROOT_CDEG_MEAN', 'ROOT_CDEG_VAR', 'ROOT_CDEG_MIN', 'ROOT_CDEG_MAX'])
    USE_ROOT_COLUMN_PCOEFS = set(['ROOT_PCOEFS_MEAN', 'ROOT_PCOEFS_VAR', 'ROOT_PCOEFS_COUNT', 'ROOT_PCOEFS_MAX', 'ROOT_PCOEFS_MIN'])
    USE_ROOT_COLUMN_NCOEFS = set(['ROOT_NCOEFS_MEAN', 'ROOT_NCOEFS_VAR', 'ROOT_NCOEFS_COUNT', 'ROOT_NCOEFS_MAX', 'ROOT_NCOEFS_MIN'])
    UNION_USE_ROOT_COLUMN = set(['USE_ROOT_CDEG', 'USE_ROOT_COLUMN_PCOEFS', 'USE_ROOT_COLUMN_NCOEFS'])

    USE_ACTIVE1 = set(['ACONS_NB1', 'ACONS_SUM1', 'ACONS_MEAN1', 'ACONS_VAR1', 'ACONS_MAX1', 'ACONS_MIN1'])
    USE_ACTIVE2 = set(['ACONS_NB2', 'ACONS_SUM2', 'ACONS_MEAN2', 'ACONS_VAR2', 'ACONS_MAX2', 'ACONS_MIN2'])
    USE_ACTIVE3 = set(['ACONS_NB3', 'ACONS_SUM3', 'ACONS_MEAN3', 'ACONS_VAR3', 'ACONS_MAX3', 'ACONS_MIN3'])
    USE_ACTIVE4 = set(['ACONS_NB4', 'ACONS_SUM4', 'ACONS_MEAN4', 'ACONS_VAR4', 'ACONS_MAX4', 'ACONS_MIN4'])
    USE_ACONS_SUM1 =set(['ACONS_SUM1', 'ACONS_VAR1'])
    USE_ACONS_SUM2 =set(['ACONS_SUM2', 'ACONS_VAR2'])
    USE_ACONS_SUM3 =set(['ACONS_SUM3', 'ACONS_VAR3'])
    USE_ACONS_SUM4 =set(['ACONS_SUM4', 'ACONS_VAR4'])
    USE_ACONS_MEAN1 = set(['ACONS_MEAN1', 'ACONS_VAR1'])
    USE_ACONS_MEAN2 = set(['ACONS_MEAN2', 'ACONS_VAR2'])
    USE_ACONS_MEAN3 = set(['ACONS_MEAN3', 'ACONS_VAR3'])
    USE_ACONS_MEAN4 = set(['ACONS_MEAN4', 'ACONS_VAR4'])
    UNION_USE_ACTIVE = set(['USE_ACTIVE1', 'USE_ACTIVE2', 'USE_ACTIVE3', 'USE_ACTIVE4'])

    USE_LB = set(["HAS_LB", "SOL_IS_AT_LB"])
    USE_UB = set(["HAS_UB", "SOL_IS_AT_UB"])
    USE_BASIS_STATUS = set(['BASIS_STATUS_0', 'BASIS_STATUS_1', 'BASIS_STATUS_2', 'BASIS_STATUS_3'])
    USE_PSEUDO = set(['PS_DOWN', 'PS_PRODUCT', 'PS_RATIO', 'PS_SUM', 'PS_UP'])

    UNION_USE_COLUMN = set(["USE_CDEG", "USE_RHS", "USE_OTA", "USE_ACTIVE"])

    USE_CDEG_MEAN = set(['CDEG_MEAN', 'CDEG_VAR', 'CDEG_MEAN_RATIO'])
    USE_CDEG_MAX = set(['CDEG_MAX', 'CDEG_MAX_RATIO'])
    USE_CDEG_MIN = set(['CDEG_MIN', 'CDEG_MIN_RATIO'])
    USE_CDEG = set(['CDEG_MIN', 'CDEG_MAX', 'CDEG_MEAN', 'CDEG_VAR', 'CDEG_MEAN_RATIO', 'CDEG_MAX_RATIO', 'CDEG_MIN_RATIO'])

    USE_RHS = set(['PRHS_RATIO_MAX', 'PRHS_RATIO_MIN', 'NRHS_RATIO_MAX', 'NRHS_RATIO_MIN'])

    USE_OTA_P = set(['OTA_PN_MAX', 'OTA_PN_MIN', 'OTA_PP_MAX', 'OTA_PP_MIN'])
    USE_OTA_N = set(['OTA_NN_MAX', 'OTA_NN_MIN', 'OTA_NP_MAX', 'OTA_NP_MIN'])
    UNION_USE_OTA = set(['USE_OTA_P', 'USE_OTA_N'])

    USE_COEFS = set(['COEF_NORMALIZED', 'COEFS'])

    UNION_SEQ = ['UNION_USE_ROOT_COLUMN', 'UNION_USE_ACTIVE', 'UNION_USE_OTA', 'UNION_USE_COLUMN']

COLUMN_FEATURES = Macros.USE_CDEG | Macros.USE_RHS | Macros.USE_OTA_P | Macros.USE_OTA_N | Macros.USE_ACTIVE1 | Macros.USE_ACTIVE2 | Macros.USE_ACTIVE3 | Macros.USE_ACTIVE4
COLUMN_FEATURES = set( x.lower() for x in COLUMN_FEATURES)
assert COLUMN_FEATURES.issubset(set(FEATURE_NAMES))

NODE_FEATURES_SIMPLE = [x for x in NODE_FEATURES if (x not in COLUMN_FEATURES)]
KHALI_FEATURES_SIMPLE = [x for x in KHALI_FEATURES if (x not in COLUMN_FEATURES)]
FEATURE_NAMES_SIMPLE = NODE_FEATURES_SIMPLE + KHALI_FEATURES_SIMPLE

NODE_FEATURE_NUM_SIMPLE, KHALI_FEATURE_NUM_SIMPLE = 19, 29
WHERE_PSEUDO_SIMPLE = FEATURE_NAMES_SIMPLE.index("ps_product")
TOTAL_FEATURE_NUM_SIMPLE = NODE_FEATURE_NUM_SIMPLE + KHALI_FEATURE_NUM_SIMPLE
assert TOTAL_FEATURE_NUM_SIMPLE == 48


def get_macros(features_origin, debug=False):
    features = set([x.upper() for x in features_origin])
    macros = []
    for ith, feature in enumerate(features):
        macros.append(f"#define {feature} {ith}\n")
    macros.append(f"#define NUM_FEATURE {len(features)}\n")

    uses = set()
    for k,v in Macros.__dict__.items():
        if k[:4] == "USE_":
            if (features & v):
                macros.append(f"#define {k}\n")
                uses.add(k)
    for k in Macros.UNION_SEQ:
        key, value = k[6:], Macros.__dict__[k]
        if (uses & value):
            uses.add(key)
            macros.append(f"#define {key}\n")

    if debug:
        for k in ["RAW", "NORMED", "SCORE"]:
            macros.append(f"#define DEBUG_{k}\n")

    return macros


@hydra.main(config_path='', config_name='eval', version_base=None)
def set_const_eval(conf):
    global GLOBAL_INFO_DICT, DEVICE
    DEVICE = torch.device("cpu")
    torch.set_default_device(DEVICE)
    GLOBAL_INFO_DICT["num_threads"] = NUM_THREADS = getattr(conf, "num_threads", 1)
    if NUM_THREADS > 1:
        print(f"use {NUM_THREADS} threads {IMPORTANT_INFO_SUFFIX}")
    assert type(NUM_THREADS) is int
    torch.set_num_threads(NUM_THREADS)
    os.environ["OPENBLAS_NUM_THREADS"] = str(NUM_THREADS)
    GLOBAL_INFO_DICT["instance_type"] = conf.instance_kwargs.instance_type


def set_const_train():
    global DEVICE
    DEVICE = torch.device("cuda:0")
    torch.set_default_device(DEVICE)

def set_const_test():
    global DEVICE
    DEVICE = torch.device("cuda:0")
    torch.set_default_device(DEVICE)

def set_const(mode):
    globals()[f"set_const_{mode}"]()
