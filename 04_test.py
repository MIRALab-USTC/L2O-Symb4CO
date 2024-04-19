import settings.consts as consts
consts.set_const("test")

import argparse
import os.path as osp

import train_utils
from utils.utils import get_expression_list_from_file
import dso_utils.expressions as expressions

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'problem',
        help='MILP instance type to process.',
        choices=['setcover', 'cauctions', 'facilities', 'indset'],
    )
    args = parser.parse_args()

    _, _, dataloader = train_utils.get_all_dataset(args.problem, dataset_type=None, get_train=False, get_valid=False, get_test=True)

    exp_str = get_expression_list_from_file(args.problem, best=True, eval_num=1)[0]
    model = expressions.ExpressionFromStr(exp_str=exp_str)
    ensembled_model = expressions.EnsemBleExpression([model])

    precision = train_utils.get_precision_iteratively(ensembled_model, dataloader).item()
    print(f"the imitation learning accuracy of {args.problem} is: {precision:.2f}")