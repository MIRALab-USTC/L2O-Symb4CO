from math import ceil
import torch
from torch.optim import Adam
import utils.logger as logger
import settings.consts as consts
import numpy as np

def normalize_meanstd(returns):
    return (returns - returns.mean()) / (returns.std() + consts.SAFE_EPSILON)

def normalize_minmax(returns):
    min_return = returns.min()
    delta = returns.max() - min_return
    if delta < consts.SAFE_EPSILON:
        delta = 1.
    return (returns - min_return) / delta

class PPOAlgo():
    def __init__(self,
                 agent,

                 lr_actor=4e-4,
                 K_epochs=8,
                 eps_clip=0.2,
                 entropy_coef=5e-3,
                 entropy_gamma=0.8,
                 entropy_decrease=False,
                 lr_decrease=False,
                 decrease_period=1000,
                 record_num_half=1,

                 new_clip=False,
                 return_norm = "minmax",
                 detailed=consts.DETAILED_LOG,
                 detailed_freq=consts.DETAILED_LOG_FREQ
                 ):
        self.agent = agent

        self.K_epochs = K_epochs
        self.clip_low, self.clip_high = 1 - eps_clip, 1 + eps_clip

        self.optimizer = Adam([
                        {'params': agent.parameters(), 'lr': lr_actor},
                    ])
        
        self.entropy_coef = entropy_coef
        self.entropy_gamma = entropy_gamma

        self.entropy_decrease, self.lr_decrease = entropy_decrease, lr_decrease
        self.decrease_period = decrease_period
        self.record_num_half = record_num_half
        
        self.new_clip = new_clip
        self.return_norm_func = globals()[f"normalize_{return_norm}"]

        self.detailed = detailed
        self.detailed_freq = detailed_freq

    def train(self, sequences, all_lengths, log_probs, index_useful, info_lists, returns, train_iter):
        torch.cuda.empty_cache()
        detailed_log = (self.detailed and (train_iter % self.detailed_freq == 0))
        if detailed_log:
            detailed_dict_list = []

        if (train_iter+1) % self.decrease_period == 0:
            if self.entropy_decrease:
                self.entropy_coef *= 0.8
            if self.lr_decrease:
                for g in self.optimizer.param_groups:
                    g["lr"] *= 0.8
        processed_advantages = self.return_norm_func(returns)

        for i in range(self.K_epochs):
            new_entropies, new_log_probs = self.agent.sample_sequence_train(sequences, info_lists)
            log_ratios = new_log_probs - log_probs

            if self.new_clip:
                raise NotImplementedError
            else:
                joint_prob_ratios = (index_useful * log_ratios).sum(dim=1)
                joint_ratios = joint_prob_ratios.exp()

                sign_advantage_index = torch.sign(processed_advantages)
                clipped_ratios = torch.min(joint_ratios * sign_advantage_index, torch.clamp(joint_ratios, self.clip_low, self.clip_high) * sign_advantage_index) * sign_advantage_index

            loss_actor = (clipped_ratios * processed_advantages).mean()

            entropy_gamma = torch.pow(self.entropy_gamma, torch.arange(new_entropies.shape[1]))[None, :]
            loss_entropy = (index_useful * new_entropies * entropy_gamma).sum(dim=1).mean()

            loss = - loss_actor - self.entropy_coef * loss_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if detailed_log and (i<self.record_num_half or i>=self.K_epochs - self.record_num_half):
                detailed_dict = dict(
                    processed_advantages=processed_advantages,
                    joint_ratios=joint_ratios,
                    clipped_ratios=clipped_ratios,
                    log_probs=log_probs,
                    all_lengths=all_lengths
                )
                detailed_dict = {k:v.detach().cpu().numpy() for k,v in detailed_dict.items()}

                detailed_dict_list.append(detailed_dict)

        ratios = log_ratios.exp()
        with torch.no_grad():
            record_dict = dict(
                loss_actor=loss_actor,
                loss_entropy=loss_entropy,
                loss=loss,
                clip_frac=((joint_ratios - clipped_ratios).abs() > consts.SAFE_EPSILON).sum() / len(joint_ratios),
                approx_KL=(((ratios-1) - log_ratios) * index_useful).sum() / all_lengths.sum(),
                average_len=torch.mean(all_lengths.float()),
                std_len=torch.std(all_lengths.float())
            )
        record_dict = {f"rl_algo/{k}":v.item() for k,v in record_dict.items()}

        if detailed_log:
            self.detailed_log(detailed_dict_list)

        return record_dict

    def detailed_log(self, detailed_dict_list):
        for ith, detailed_dict in enumerate(detailed_dict_list):
            logger.log_hist({f"rl_algo/{ith}th_epoch/processed_advantages": detailed_dict["processed_advantages"],
                                f"rl_algo/{ith}th_epoch/joint_ratios": detailed_dict["joint_ratios"],
                                f"rl_algo/{ith}th_epoch/clipped_ratios": detailed_dict["clipped_ratios"],
                                })
