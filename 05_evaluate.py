import settings.consts as consts
consts.set_const("eval")

from os import path as osp
import hydra
from omegaconf import OmegaConf
from torch.multiprocessing import set_start_method

import utils.utils as utils
import utils.logger as logger
import evaluate_utils


@hydra.main(config_path='settings', config_name='eval', version_base=None)
def main(conf):
    instance_type = conf.instance_kwargs.instance_type
    utils.initial_logdir_and_get_seed(exp_type="evaluate", instance_type=instance_type, exp_name=("debug" if getattr(conf, "debug", False) else conf.exp_name), dataset_type=conf.instance_kwargs.dataset_type)

    if conf.exp_name == "valid":
        conf.instance_kwargs.dataset_prefix = "valid"
        conf.instance_kwargs.n_instance = 4
        agent = evaluate_utils.TestAgent(**OmegaConf.to_container(conf.eval_kwargs), instance_kwargs=OmegaConf.to_container(conf.instance_kwargs))

        exp_list = utils.get_expression_list_from_file(instance_type)
        best_time = best_gap = float("inf")
        for i, expression in enumerate(exp_list):
            result = agent.evaluate(f"{i}:{utils.get_nlp_from_expression(expression)[0]}", expression)[0]
            valid_time, valid_gap = result["stime"], result["gap"]
            if valid_time < best_time or ( valid_time == best_time and valid_gap < best_gap):
                best_expression = expression
                best_time, best_gap = valid_time, valid_gap
        with open(osp.join(consts.EXPRESSION_DIR, f"{instance_type}_best"), "w") as txt:
            txt.write(best_expression)
        logger.log(f"Best expression: {best_expression}")

    else:
        assert conf.exp_name == "default" and conf.instance_kwargs.dataset_prefix == "transfer"
        agent = evaluate_utils.TestAgent(**OmegaConf.to_container(conf.eval_kwargs), instance_kwargs=OmegaConf.to_container(conf.instance_kwargs))
        
        expression = utils.get_expression_list_from_file(instance_type, best=True, eval_num=1)[0]
        agent.evaluate(f"{utils.get_nlp_from_expression(expression)[0]}", expression)
        agent.evaluate()


if __name__ == "__main__":
    set_start_method('spawn')
    main()
