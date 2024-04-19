import settings.consts as consts
consts.set_const("train")

from os import path as osp
import hydra
from omegaconf import OmegaConf

from train_utils import TrainDSOAgent
import utils.utils as utils


@hydra.main(config_path='settings', config_name='train', version_base=None)
def main(conf):
    log_dir = utils.get_log_dir(exp_type="train", instance_type=conf.instance_kwargs.instance_type, exp_name=conf.exp_name)
    continue_train_path = getattr(conf, "continue_train_path", None)
    for i in range(conf.exp_num):
        new_conf = OmegaConf.to_container(conf, resolve=True)
        train_kwargs, instance_kwargs, expression_kwargs, dso_agent_kwargs, rl_algo_kwargs = new_conf["train_kwargs"], new_conf["instance_kwargs"], new_conf["expression_kwargs"], new_conf["dso_agent_kwargs"], new_conf["rl_algo_kwargs"]

        logdir, _ = utils.initial_logger_and_seed(log_dir, i, new_conf)
        train_agent = TrainDSOAgent(**train_kwargs, continue_train_path=continue_train_path, instance_kwargs=instance_kwargs, expression_kwargs=expression_kwargs, dso_agent_kwargs=dso_agent_kwargs, rl_algo_kwargs=rl_algo_kwargs)
        train_agent.process()
        del train_agent

        expression_save_path, logdir_record_path = osp.join(consts.EXPRESSION_DIR, conf.instance_kwargs.instance_type), osp.join(logdir, "best.txt")
        with open(expression_save_path, "a") as expression_save_file, open(logdir_record_path, "r") as logdir_record_file:
            lines = logdir_record_file.readlines()

            ith = 0
            expression_save_file.write(f"\n{logdir}:\n")
            for line in reversed(lines):
                assert line[:9] == "iteration"
                if ith < 10:
                    ith += 1
                    expression_save_file.write(line)
                if ith >= 10:
                    break

if __name__ == "__main__":
    main()
