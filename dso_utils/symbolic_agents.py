"""
File adapted from https://github.com/dandip/DSRPytorch
"""


import torch.nn as nn
import torch.nn.functional as F
import torch

from settings.consts import SAFE_EPSILON

class DSOAgent(nn.Module):
    def __init__(self, operators, min_length=4, max_length=64, hidden_size=128, num_layers=2, soft_length=20, two_sigma_square=16):
        super().__init__()

        self.input_size = 2 * operators.operator_length # One-hot encoded parent and sibling
        self.hidden_size = hidden_size
        self.output_size = operators.operator_length # Output is a softmax distribution over all operators
        self.num_layers = num_layers
        self.operators = operators

        # Initial cell optimization
        self.init_input = nn.Parameter(data=torch.rand(1, self.input_size), requires_grad=True)
        self.init_hidden = nn.Parameter(data=torch.rand(self.num_layers, 1, self.hidden_size), requires_grad=True)

        self.min_length = min_length
        self.max_length = max_length
        self.soft_length = soft_length
        self.two_sigma_square = two_sigma_square

        self.lstm = nn.LSTM(
            input_size = self.input_size,
            hidden_size = self.hidden_size,
            num_layers = self.num_layers,
            proj_size = self.output_size,
        )
        self.init_hidden_lstm = nn.Parameter(data=torch.rand(self.num_layers, 1, self.output_size), requires_grad=True)
        self.activation = nn.Softmax(dim=1)

    @torch.no_grad()
    def sample_sequence_eval(self, n):
        sequences = torch.zeros(n, 0, dtype=torch.long)

        input_tensor = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        sequence_mask = torch.ones(n, dtype=torch.bool)
        counters = torch.ones(n, 1, dtype=torch.long) # Number of tokens that must be sampled to complete expression

        length = 0
        all_lengths = torch.zeros(n, dtype=torch.long)

        # While there are still tokens left for sequences in the batch
        all_log_prob_list, all_counters_list, all_inputs_list = [], [], []
        while(sequence_mask.any()):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, length)

            # Apply constraints and normalize distribution
            output = self.apply_constraints(output, counters, length, sequences)
            output = output / torch.sum(output, dim=1, keepdim=True)

            # Sample from categorical distribution
            dist = torch.distributions.Categorical(output)
            token = dist.sample()

            # Add sampled tokens to sequences
            sequences = torch.cat((sequences, token[:, None]), dim=1)
            length += 1
            all_lengths[sequence_mask] += 1

            # Add log probability of current token
            all_log_prob_list.append(dist.log_prob(token)[:, None])

            # Add entropy of current token
            all_counters_list.append(counters)
            all_inputs_list.append(input_tensor)

            # Update counter
            counters = counters + (torch.logical_and(self.operators.arity_one_begin<=token, token<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=token, token<self.operators.arity_two_end).long() * 2 - 1)[:, None]
            sequence_mask = torch.logical_and(sequence_mask, counters.squeeze(1) > 0)

            # Compute next parent and sibling; assemble next input tensor
            input_tensor = self.get_parent_sibling(n, sequences, length-1, sequence_mask)

        # Filter entropies log probabilities using the sequence_mask
        assert all_lengths.min() >= self.min_length and all_lengths.max() <= self.max_length+1 and all_lengths.max() == sequences.shape[1]
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return sequences, all_lengths, log_probs, (all_counters_list, all_inputs_list)

    def sample_sequence_train(self, sequences, info_lists):
        all_counters_list, all_inputs_list = info_lists
        assert sequences.shape[1] == len(all_counters_list) == len(all_inputs_list)
        n = len(sequences)

        all_inputs_list[0] = self.init_input.expand(n, -1).contiguous()
        hidden_tensor = self.init_hidden.expand(-1, n, -1).contiguous()
        hidden_lstm = self.init_hidden_lstm.expand(-1, n, -1).contiguous()

        all_log_prob_list, all_entropy_list = [], []
        for i, (token, counters, input_tensor) in enumerate(zip(sequences.t(), all_counters_list, all_inputs_list)):
            output, hidden_tensor, hidden_lstm = self.forward(input_tensor, hidden_tensor, hidden_lstm, i)

            output = self.apply_constraints(output, counters, i, sequences[:,:i])
            output = output / torch.sum(output, dim=1, keepdim=True)

            dist = torch.distributions.Categorical(output)
            all_log_prob_list.append(dist.log_prob(token)[:, None])
            all_entropy_list.append(dist.entropy()[:, None])

        entropies = torch.cat(all_entropy_list, dim=1)
        log_probs = torch.cat(all_log_prob_list, dim=1)

        return entropies, log_probs


    def forward(self, input, hidden, hidden_lstm, cur_length):
        """Input should be [parent, sibling]
        """

        output, (hn, cn) = self.lstm(input.unsqueeze(0).float(), (hidden_lstm, hidden))
        output = output.squeeze(0)

        # ~ soft length constraint
        prior_vec = torch.zeros(self.output_size)
        # cur_length = sequences.shape[1]
        if cur_length < self.soft_length:
            prior_vec[self.operators.arity_zero_begin:self.operators.arity_zero_end] = - (self.soft_length - cur_length) ** 2 / self.two_sigma_square
        elif cur_length > self.soft_length:
            prior_vec[self.operators.arity_two_begin:self.operators.arity_two_end] = - (cur_length - self.soft_length) ** 2 / self.two_sigma_square
        output = output + prior_vec[None, :]

        output = self.activation(output)
        return output, cn, hn

    def apply_constraints(self, output, counters, length, sequences):
        """Applies in situ constraints to the distribution contained in output based on the current tokens
        """
        # Add small epsilon to output so that there is a probability of selecting
        # everything. Otherwise, constraints may make the only operators ones
        # that were initially set to zero, which will prevent us selecting
        # anything, resulting in an error being thrown
        
        output = output + SAFE_EPSILON

        # turn off column features if we set var_list = simple
        if self.operators.column_var_mask is not None:
            output[:, self.operators.column_var_mask] = 0.

        # ~ Check that minimum length will be met ~
        # Explanation here
        min_boolean_mask = (counters + length) >= self.min_length
        min_length_mask = torch.logical_or(self.operators.nonzero_arity_mask, min_boolean_mask)
        output[~min_length_mask] = 0

        # ~ Check that maximum length won't be exceed ~
        max_boolean_mask = (counters + length) < self.max_length
        max_length_mask = torch.logical_or(self.operators.zero_arity_mask, max_boolean_mask)
        output[~max_length_mask] = 0

        # forbid direct inverse function
        # last_token = sequences[:, -1]
        # output[xxx, ]


        # ~ Ensure that all expressions have a variable ~
        # nonvar_zeroarity_mask = ~torch.logical_and(self.operators.zero_arity_mask, self.operators.nonvariable_mask)
        if (length == 0): # First thing we sample can't be
            # output = torch.minimum(output, nonvar_zeroarity_mask)
            output[:, self.operators.const_mask.squeeze(0)] = 0
        else:
            last_counter_mask = (counters == 1)
            non_var_now_mask = torch.logical_not( (sequences < self.operators.variable_end).any(dim=1, keepdim=True) )
            last_token_and_no_var_mask = torch.logical_and(last_counter_mask, non_var_now_mask)
            const_and_last_token_and_no_var_mask = torch.logical_and(last_token_and_no_var_mask, self.operators.const_mask)
            output[const_and_last_token_and_no_var_mask] = 0

            # ~ forbid inverse unary
            last_token = sequences[:, -1]
            last_token_has_inverse = torch.where(self.operators.have_inverse[last_token])[0]
            last_token_inverse = self.operators.where_inverse[last_token[last_token_has_inverse]]
            output[last_token_has_inverse, last_token_inverse] = 0

        return output

    def get_parent_sibling(self, batch_size, sequences, recent, sequence_mask):
        """Returns parent, sibling for the most recent token in token_list
        """
        parent_sibling = torch.full(size=(batch_size, 2), fill_value=-1, dtype=torch.long)
        parent_sibling[~sequence_mask] = 0

        token_last = sequences[:, recent]
        token_last_is_parent = self.operators.arity_tensor[token_last] > 0
        parent_sibling[token_last_is_parent, 0] = token_last[token_last_is_parent]
        parent_sibling[token_last_is_parent, 1] = self.output_size # means empty token

        c = torch.zeros(batch_size, dtype=torch.long)
        for i in range(recent, -1, -1):
            # Determine arity of the i-th tokens
            unfinished_bool_index = (parent_sibling < 0).any(dim=1)
            if not unfinished_bool_index.any():
                break

            unfinished_token_i = sequences[:, i][unfinished_bool_index]

            # Increment c by arity of the i-th token, minus 1
            c[unfinished_bool_index] += (torch.logical_and(self.operators.arity_one_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_one_end).long() \
                        + torch.logical_and(self.operators.arity_two_begin<=unfinished_token_i, unfinished_token_i<self.operators.arity_two_end).long() * 2 - 1)

            # In locations where c is zero (and parents and siblings that are -1),
            # we want to set parent_sibling to sequences[:, i] and sequeneces[:, i+1].
            # c_mask an n x 1 tensor that is True when c is zero and parents/siblings are -1.
            # It is False otherwise.
            found_now = torch.logical_and(unfinished_bool_index, c==0)

            # n x 2 tensor where dimension is 2 is sequences[:, i:i+1]
            # Since i+1 won't exist on the first iteration, we pad
            # (-1 is i+1 doesn't exist)
            parent_sibling[found_now] = sequences[found_now, i:(i+2)]

            # Set i_ip1 to 0 for indices where c_mask is False
            # i_ip1 = i_ip1 * c_mask.long()

            # Set parent_sibling to 0 for indices where c_mask is True
            # parent_sibling = parent_sibling * (~c_mask).long()

            # parent_sibling = parent_sibling + i_ip1


        assert (parent_sibling >= 0).all()

        parent_sibling_where_empty = (parent_sibling == self.output_size)
        parent_sibling[parent_sibling_where_empty] = 0

        input_tensor = F.one_hot(parent_sibling.reshape(-1), num_classes=self.output_size)  #())
        input_tensor[parent_sibling_where_empty.reshape(-1)] = 0.
        input_tensor = input_tensor.reshape(batch_size, self.input_size)


        return input_tensor
