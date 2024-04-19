"""
File adapted from https://github.com/dandip/DSRPytorch
"""

import torch
from collections import OrderedDict, defaultdict

import settings.consts as consts
import utils.logger as logger

MATH_ARITY = OrderedDict([ # arity 2 -> arity 1
    ('+',2),
    ('-',2),
    ('*',2),
    ('/',2), 
    ('^', 2),
    ('exp',1),
    ('log',1),
])

CONSTANT_OPERATORS = ["2.0", "5.0", "10.0", "0.1", "0.2", "0.5"] # @ means a place holder which will be optimized in the inner loop
INVERSE_OPERATOR_DICT = {"exp": "log", "log": "exp", "sqrt": "square", "square": "sqrt"}

def binary_f(x,a,b):
    return f"({a} {x} {b})"
def unary_f(x,a):
    return f"torch.{x}({a})"
def unary_f_nlp(x,a):
    return f"{x}({a})"
def nlp_power(x,a,b):
    return f"{a}^{b}"
def nlp_square(x,a):
    return f"{a}^2"
def nlp_sqrt(x,a):
    return f"{a}^0.5"

def power(x,a,b):
    return f"torch.pow({a}, {b})"

TORCH_OPERATOR_DICT = {
    "^": power,
}

NLP_OPERATOR_DICT = {
    "^": nlp_power,
}


class Operators:

    def __init__(self, const_list=None, math_list="all", var_list="full"):
        """
        order: vars, consts, arity_two_operators, arity_one_operators
        """
        self.var_operators = consts.FEATURE_NAMES
        if var_list=="simple":
            column_var_mask = [i for i, name in enumerate(consts.FEATURE_NAMES) if name in consts.COLUMN_FEATURES]
            self.column_var_mask = torch.tensor(column_var_mask, dtype=torch.long)
            assert len(self.column_var_mask) == 91 - 48
        else:
            assert var_list=="full"
            self.column_var_mask = None

        if const_list is None:
            self.constant_operators = CONSTANT_OPERATORS[:]
        else:
            self.constant_operators = const_list
        if math_list == "simple":
            self.math_operators = ['+', '-', '*']
        elif math_list == "all":
            self.math_operators = list(MATH_ARITY.keys())
        elif type(math_list) is list:
            math_set = set(math_list)
            self.math_operators = [x for x in MATH_ARITY if x in math_set]
        else:
            raise NotImplementedError(f"wrong math operators list {consts.WARN_INFO_SUFFIX}")

        self.operator_list = self.var_operators + self.constant_operators + self.math_operators
        self.operator_dict = {k:i for i,k in enumerate(self.operator_list)}
        self.operator_length = len(self.operator_list)

        arity_dict = defaultdict(int, MATH_ARITY)
        self.arity_list = [arity_dict[operator] for operator in self.operator_list]
        self.arity_tensor = torch.tensor(self.arity_list, dtype=torch.long)

        self.zero_arity_mask = torch.tensor([True if arity_dict[x]==0 else False for x in self.operator_list], dtype=torch.bool)[None, :]
        self.nonzero_arity_mask = torch.tensor([True if arity_dict[x]!=0 else False for x in self.operator_list], dtype=torch.bool)[None, :]

        self.have_inverse = torch.tensor([((operator in INVERSE_OPERATOR_DICT) and (INVERSE_OPERATOR_DICT[operator] in self.operator_dict)) for operator in self.operator_list], dtype=torch.bool)
        self.where_inverse = torch.full(size=(self.operator_length,), fill_value=int(1e5), dtype=torch.long)
        self.where_inverse[self.have_inverse] = torch.tensor([self.operator_dict[INVERSE_OPERATOR_DICT[operator]] for i, operator in enumerate(self.operator_list) if self.have_inverse[i]], dtype=torch.long)

        variable_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        variable_mask[:len(self.var_operators)] = True
        const_mask = torch.zeros(len(self.operator_list), dtype=torch.bool)
        const_mask[len(self.var_operators):-len(self.math_operators)] = True
        self.variable_mask = variable_mask[None, :]
        self.non_variable_mask = torch.logical_not(self.variable_mask)
        self.const_mask = const_mask[None, :]
        self.non_const_mask = torch.logical_not(self.const_mask)

        num_math_arity_two = sum([1 for x in self.math_operators if MATH_ARITY[x]==2])
        num_math_arity_one = len(self.math_operators) - num_math_arity_two
        self.arity_zero_begin, self.arity_zero_end = 0, len(self.var_operators) + len(self.constant_operators)
        self.arity_two_begin, self.arity_two_end = len(self.var_operators) + len(self.constant_operators), len(self.var_operators) + len(self.constant_operators) + num_math_arity_two
        self.arity_one_begin, self.arity_one_end = len(self.operator_list) - num_math_arity_one, len(self.operator_list)

        self.variable_begin, self.variable_end = 0, len(self.var_operators)


    def is_var_i(self, i):
        return self.variable_begin <= i < self.variable_end
