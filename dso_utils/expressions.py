"""
File adapted from https://github.com/dandip/DSRPytorch
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from contextlib import contextmanager

import dso_utils.operators as operators_module
import settings.consts as consts

@contextmanager
def set_train_mode(network, train_mode=True):
    if train_mode:
        yield
    else:
        with torch.no_grad():
            network.eval()
            yield
            network.train()

class OperatorNode:
    def __init__(self, operator, operator_str, arity, is_var, parent=None):
        """Description here
        """
        self.operator = operator
        self.operator_str = operator_str
        self.arity = arity
        self.is_var = is_var
        self.parent = parent
        self.left_child = None
        self.right_child = None

    def add_child(self, node):
        if (self.left_child is None):
            self.left_child = node
        elif (self.right_child is None):
            self.right_child = node
        else:
            raise RuntimeError("Both children have been created.")

    def set_parent(self, node):
        self.parent = node

    def remaining_children(self):
        if (self.arity == 0):
            return False
        elif (self.arity == 1 and self.left_child is not None):
            return False
        elif (self.arity == 2 and self.left_child is not None and self.right_child is not None):
            return False
        return True

    def print(self, operator_dict, unary_func, use_tensor):
        if (self.arity == 2):
            left_print = self.left_child.print(operator_dict, unary_func, use_tensor)
            right_print = self.right_child.print(operator_dict, unary_func, use_tensor)
            return operator_dict.get(self.operator_str, operators_module.binary_f)(self.operator_str, left_print, right_print)
        elif (self.arity == 1):
            left_print = self.left_child.print(operator_dict, unary_func, use_tensor)
            value = operator_dict.get(self.operator_str, unary_func)(self.operator_str, left_print)
            return value
        else:
            assert self.arity == 0
            if self.is_var:
                return f"inputs[:,{self.operator}]" if use_tensor else self.operator_str
            else:
                if use_tensor:
                    return f"torch.tensor({self.operator_str}, dtype=torch.float, device=consts.DEVICE)"
                else:
                    return self.operator_str


def construct_tree(operator_list, arity_list, variable_end, sequence):
    root = OperatorNode(sequence[0], operator_list[sequence[0]], arity_list[sequence[0]], sequence[0] < variable_end)
    past_node = root
    for operator in sequence[1:]:
        curr_node = OperatorNode(operator, operator_list[operator], arity_list[operator], operator < variable_end, parent=past_node)
        past_node.add_child(curr_node)
        past_node = curr_node
        while not past_node.remaining_children():
            past_node = past_node.parent
            if (past_node is None):
                assert operator == sequence[-1]
                break
    return root


class Expression(nn.Module):
    def __init__(self, sequence, operators):
        super().__init__()
        self.sequence = sequence
        self.root = construct_tree(operators.operator_list, operators.arity_list, operators.variable_end, sequence)
        self.expression = self.root.print(operators_module.TORCH_OPERATOR_DICT, operators_module.unary_f, True)

    def get_nlp(self):
        return self.root.print(operators_module.NLP_OPERATOR_DICT, operators_module.unary_f_nlp, False)

    def get_expression(self):
        expression = self.expression
        return expression

    def forward(self, inputs):
        return F.tanh(eval(self.expression))

class ExpressionFromStr(nn.Module):
    def __init__(self, exp_str):
        super().__init__()
        self.expression = exp_str

    def forward(self, inputs):
        return F.tanh(eval(self.expression))

class EnsemBleExpression(nn.Module):
    def __init__(self, models):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.ensemble_num = len(self.models)
        self.train()

    def forward(self, x, train_mode=True):
        with set_train_mode(self, train_mode):
            futures = [torch.jit.fork(model, x) for model in self.models]
            results = [torch.jit.wait(fut) for fut in futures]
            return torch.stack(results)
