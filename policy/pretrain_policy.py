import tensorflow as tf
import numpy as np
from policy import base_policy
from util import tf_util, network_util
from util.misc_util import generate_haar
from collections import OrderedDict

class SparsePPOPolicy(base_policy.BasePolicy):
    def __init__(self, params, session, scope,
                 observation_size, action_size, task_names_list, *args, **kwargs):
        super(SparsePPOPolicy, self).__init__(params, session,
            scope, observation_size, action_size)

        self.required_keys = ['start_state', 'end_state', 'action', 'rewards', 'returns',
                                'old_action_dist_mu', 'old_action_dist_logstd']

        self.task_names = task_names_list

    def build_model(self):
        with tf.variable_scope(self._name_scope):
            self._build_ph()

            self._tensor = {}

            # Important parameters
            self._ppo_clip = self.params.ppo_clip
            self._kl_eta = self.params.kl_eta
            self._current_kl_lambda = 1
            self._current_lr = self.params.policy_lr
            self._timesteps_so_far = 0

            # construct the input to the forward network, we normalize the state
            # input, and concatenate with the action
            self._tensor['normalized_start_state'] = (
                self._input_ph['start_state'] -
                self._whitening_operator['state_mean']
            ) / self._whitening_operator['state_std']
            self._tensor['net_input'] = self._tensor['normalized_start_state']
            # the mlp for policy
            network_shape = [self._observation_size] + \
                self.params.policy_network_shape + [self._action_size]
            num_layer = len(network_shape) - 1
            act_type = \
                [self.params.policy_activation_type] * (num_layer - 1) + [None]
            norm_type = \
                [self.params.policy_normalizer_type] * (num_layer - 1) + [None]
            init_data = []
            for _ in range(num_layer):
                init_data.append(
                    {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                     'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
                )
            init_data[-1]['w_init_para']['stddev'] = 0.01  # the output layer std
            self._MLP = network_util.SparseMLP(
                dims=network_shape, scope='policy_mlp', train=True,
                activation_type=act_type, normalizer_type=norm_type,
                init_data=init_data, linear_last_layer=True
            )
            self._input_ph['policy_sparse_masks'] = self._MLP._sparse_mask
            self._input_ph['policy_embed'] = self._MLP._embed
            self._tensor['policy_weights'] = self._MLP._w
            self._tensor['policy_b'] = self._MLP._b

            # the output policy of the network
            self._tensor['action_dist_mu'] = self._MLP(self._tensor['net_input'])
            self._tensor['action_logstd'] = tf.Variable(
                (0 * self._npr.randn(1, self._action_size)).astype(np.float32),
                name="action_logstd", trainable=True
            )

            self._tensor['action_dist_logstd'] = tf.tile(
                self._tensor['action_logstd'],
                [tf.shape(self._tensor['action_dist_mu'])[0], 1]
            )
            # make sure the size is matched to [batch, num_action]
            # fetch all the trainable variables

    def build_loss(self):
        with tf.variable_scope(self._name_scope):
            self._update_operator = {}
            self._build_value_network_and_loss()
            self._set_var_list()
            self._build_ppo_loss_preprocess()
            self._build_ppo_loss()

            self._init_sparse_params()

    def _build_value_network_and_loss(self):
        # build the placeholder for training the value function
        self._input_ph['value_target'] = \
            tf.placeholder(tf.float32, [None, 1], name='value_target')

        self._input_ph['old_values'] = \
            tf.placeholder(tf.float32, [None, 1], name='old_value_est')

        # build the baseline-value function
        network_shape = [self._observation_size] + \
            self.params.value_network_shape + [1]
        num_layer = len(network_shape) - 1
        act_type = \
            [self.params.value_activation_type] * (num_layer - 1) + [None]
        norm_type = \
            [self.params.value_normalizer_type] * (num_layer - 1) + [None]
        init_data = []
        for _ in range(num_layer):
            init_data.append(
                {'w_init_method': 'normc', 'w_init_para': {'stddev': 1.0},
                 'b_init_method': 'constant', 'b_init_para': {'val': 0.0}}
            )
        self._value_MLP = network_util.SparseMLP(
            dims=network_shape, scope='value_mlp', train=True,
            activation_type=act_type, normalizer_type=norm_type,
            init_data=init_data, linear_last_layer=True
        )

        self._input_ph['value_sparse_masks'] = self._value_MLP._sparse_mask
        self._input_ph['value_embed'] = self._value_MLP._embed
        self._tensor['value_weights'] = self._value_MLP._w
        self._tensor['value_b'] = self._value_MLP._b

        self._tensor['pred_value'] = self._value_MLP(self._tensor['net_input'])
        # build the loss for the value network
#        self._tensor['val_clipped'] = \
#            self._input_ph['old_values'] + tf.clip_by_value(
#            self._tensor['pred_value'] -
#            self._input_ph['old_values'],
#            -self._ppo_clip, self._ppo_clip)
#        self._tensor['val_loss_clipped'] = tf.square(
#            self._tensor['val_clipped'] - self._input_ph['value_target']
#        )
#        self._tensor['val_loss_unclipped'] = tf.square(
#            self._tensor['pred_value'] - self._input_ph['value_target']
#        )
#
#
#        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
#            tf.maximum(self._tensor['val_loss_clipped'],
#                       self._tensor['val_loss_unclipped'])
#        )

        self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
            tf.square(
                self._tensor['pred_value'] - self._input_ph['value_target']
            )
        )

        self._update_operator['vf_update_op'] = tf.train.AdamOptimizer(
            learning_rate=self.params.value_lr,
            beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['vf_loss'])

    def _build_ppo_loss_preprocess(self):
        # proximal policy optimization
        self._input_ph['action'] = tf.placeholder(
            tf.float32, [None, self._action_size],
            name='action_sampled_in_rollout'
        )
        self._input_ph['advantage'] = tf.placeholder(
            tf.float32, [None, 1], name='advantage_value'
        )
        self._input_ph['old_action_dist_mu'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='old_act_dist_mu'
        )
        self._input_ph['old_action_dist_logstd'] = tf.placeholder(
            tf.float32, [None, self._action_size], name='old_act_dist_logstd'
        )
        self._input_ph['batch_size'] = tf.placeholder(
            tf.float32, [], name='batch_size_float'
        )
        self._input_ph['lr'] = tf.placeholder(
            tf.float32, [], name='learning_rate'
        )
        self._input_ph['kl_lambda'] = tf.placeholder(
            tf.float32, [], name='learning_rate'
        )

        # the kl and ent of the policy
        self._tensor['log_p_n'] = tf_util.gauss_log_prob(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd'],
            self._input_ph['action']
        )

        self._tensor['log_oldp_n'] = tf_util.gauss_log_prob(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._input_ph['action']
        )

        self._tensor['ratio'] = \
            tf.exp(self._tensor['log_p_n'] - self._tensor['log_oldp_n'])

        self._tensor['ratio_clipped'] = tf.clip_by_value(
            self._tensor['ratio'], 1. - self._ppo_clip, 1. + self._ppo_clip
        )

        # the kl divergence between the old and new action
        self._tensor['kl'] = tf_util.gauss_KL(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']

        self._tensor['ent'] = tf_util.gauss_ent(
            self._tensor['action_dist_mu'],
            self._tensor['action_dist_logstd']
        ) / self._input_ph['batch_size']

    def _build_ppo_loss(self):
        self._update_operator['pol_loss_unclipped'] = \
            -self._tensor['ratio'] * \
            tf.reshape(self._input_ph['advantage'], [-1])

        self._update_operator['pol_loss_clipped'] = \
            -self._tensor['ratio_clipped'] * \
            tf.reshape(self._input_ph['advantage'], [-1])

        self._update_operator['surr_loss'] = tf.reduce_mean(
            tf.maximum(self._update_operator['pol_loss_unclipped'],
                       self._update_operator['pol_loss_clipped'])
        )

        self._update_operator['loss'] = self._update_operator['surr_loss']

        if self.params.use_weight_decay:
            self._update_operator['weight_decay_loss'] = \
                tf_util.l2_loss(self._trainable_var_list)
            self._update_operator['loss'] += \
                self._update_operator['weight_decay_loss'] * \
                self.params.weight_decay_coefficient

        # self._update_operator['optimizer'] = tf.train.AdamOptimizer(
        #          learning_rate=self._input_ph['lr'],
        #          beta1=0.5, beta2=0.99, epsilon=1e-4
        # )

        self._update_operator['policy_gradients'] = {
            self._input_ph['policy_sparse_masks'][i]:
            tf.gradients(self._update_operator['surr_loss'], self._MLP._w[i]) \
            for i in range(len(self._input_ph['policy_sparse_masks']))}

        self._update_operator['value_gradients'] = {
            self._input_ph['value_sparse_masks'][i]:
            tf.gradients(self._update_operator['vf_loss'], self._value_MLP._w[i]) \
            for i in range(len(self._input_ph['value_sparse_masks']))}

        self._update_operator['update_op'] = tf.train.AdamOptimizer(
             learning_rate=self._input_ph['lr'],
        #     # beta1=0.5, beta2=0.99, epsilon=1e-4
        ).minimize(self._update_operator['surr_loss'])

        # the actual parameter values
        self._update_operator['get_flat_param'] = \
            tf_util.GetFlat(self._session, self._trainable_var_list)
        # deprecated dont use
        # call this to set parameter values
        self._update_operator['set_from_flat_param'] = \
            tf_util.SetFromFlat(self._session, self._trainable_var_list)

    def train(self, data_dict, replay_buffer, training_info={}):

        self._generate_advantage(data_dict)
        stats = {'surr_loss': [], 'entropy': [], 'kl': [], 'vf_loss': []}
        update_kl_mean = -.1

        self._iters_so_far += 1

        temp_key = [key for key in self.task_names][0]
        num_tasks = len(self.task_names)

        self._timesteps_so_far += len(data_dict[temp_key]['start_state']) * num_tasks
        total_batch_len = len(data_dict[temp_key]['start_state'])
        total_batch_inds = np.arange(total_batch_len)

        for epoch in range(max(self.params.policy_epochs,
                               self.params.value_epochs)):
            self._npr.shuffle(total_batch_inds)
            minibatch_size = total_batch_len // self.params.num_minibatches

            for start in range(self.params.num_minibatches):
                start = start * minibatch_size
                end = min(start + minibatch_size, total_batch_len)
                batch_inds = total_batch_inds[start:end]

                task_names_rotated = self._npr.permutation(self.task_names)
                for tn in task_names_rotated:
                    feed_dict = {
                        self._input_ph[key]: data_dict[tn][batch_inds]
                        for key in ['start_state', 'action', 'advantage',
                                    'old_action_dist_mu', 'old_action_dist_logstd']
                    }

                    feed_dict = {**self._default_policy_masks[tn],
                        **self._default_policy_embed[tn],
                        **self._default_value_masks[tn], **self._default_value_embed[tn],
                        ** feed_dict}

                    feed_dict[self._input_ph['batch_size']] = \
                        np.array(float(end - start))

                    feed_dict[self._input_ph['lr']] = self._current_lr

                    if epoch < self.params.policy_epochs:
                        loss, kl, policy_gradients, _ = self._session.run(
                            [self._update_operator['surr_loss'], self._tensor['kl'],
                             self._update_operator['policy_gradients'],
                             self._update_operator['update_op']],
                            feed_dict
                        )
                        # kl = self._session.run([self._tensor['kl']], feed_dict)
                        stats['surr_loss'].append(loss)
                        stats['kl'].append(kl)

                        # train the baseline function
                        feed_dict[self._input_ph['old_values']] = \
                            data_dict[tn]['value'][batch_inds]
                        feed_dict[self._input_ph['value_target']] = \
                            data_dict[tn]['value_target'][batch_inds]
                        vf_loss, value_gradients, _ = self._session.run(
                            [self._update_operator['vf_loss'],
                             self._update_operator['value_gradients'],
                             self._update_operator['vf_update_op']],
                            feed_dict=feed_dict
                        )
                        stats['vf_loss'].append(vf_loss)

                        self.update_masks(policy_gradients, value_gradients, tn)

        # run final feed_dict with whole batch

        kl_total = 0
        for tn in self.task_names:
            feed_dict = {
                self._input_ph[key]: data_dict[tn][key]
                for key in ['start_state', 'action', 'advantage',
                            'old_action_dist_mu', 'old_action_dist_logstd']
            }

            feed_dict = {**self._default_policy_masks[tn], **self._default_policy_embed[tn],
                           **self._default_value_masks[tn], **self._default_value_embed[tn],
                         **feed_dict}
            feed_dict[self._input_ph['batch_size']] = \
                np.array(float(total_batch_len))

            kl_total += self._session.run([self._tensor['kl']], feed_dict)

        kl_total /= len(self.task_names)

        self._update_adaptive_parameters(update_kl_mean, kl_total)
        if self.params.use_kl_penalty:
            self._current_kl_lambda = self._current_kl_lambda
        # update the whitening variables
        self._set_whitening_var(data_dict['whitening_stats'])

        for key in stats:
            stats[key] = np.mean(stats[key])

        stats['advantage'] = np.mean(data_dict['advantage'])
        stats['mean_value'] = np.mean(data_dict['value'])

        return stats, data_dict

    def _update_adaptive_parameters(self, kl_epoch, i_kl_total):
        # update the lambda of kl divergence
        if self.params.use_kl_penalty:
            if kl_epoch > self.params.target_kl_high * self.params.target_kl:
                self._current_kl_lambda *= self.params.kl_alpha
                if self._current_kl_lambda > 30 and \
                        self._current_lr > 0.1 * self.params.policy_lr:
                    self._current_lr /= 1.5
            elif kl_epoch < self.params.target_kl_low * self.params.target_kl:
                self._current_kl_lambda /= self.params.kl_alpha
                if self._current_kl_lambda < 1 / 30 and \
                        self._current_lr < 10 * self.params.policy_lr:
                    self._current_lr *= 1.5

            self._current_kl_lambda = max(self._current_kl_lambda, 1 / 35.0)
            self._current_kl_lambda = min(self._current_kl_lambda, 35.0)

        # update the lr
        elif self.params.policy_lr_schedule == 'adaptive':
            mean_kl = i_kl_total
            if mean_kl > self.params.target_kl_high * self.params.target_kl:
                self._current_lr /= self.params.policy_lr_alpha
            if mean_kl < self.params.target_kl_low * self.params.target_kl:
                self._current_lr *= self.params.kl_alpha

            self._current_lr = max(self._current_lr, 3e-10)
            self._current_lr = min(self._current_lr, 1e-2)

        else:
            self._current_lr = self.params.policy_lr * max(
                1.0 - float(self._timesteps_so_far) / self.params.max_timesteps,
                0.0
            )

    def act(self, data_dict, task_name, *params):
        policy_masks = self._default_policy_masks[task_name]
        value_masks = self._default_value_masks[task_name]
        policy_embed = self._default_policy_embed[task_name]
        value_embed = self._default_value_embed[task_name]

        action_dist_mu, action_dist_logstd = self._session.run(
            [self._tensor['action_dist_mu'], self._tensor['action_dist_logstd']],
            feed_dict={**policy_masks, **policy_embed,
                       **value_masks, **value_embed,
                       self._input_ph['start_state']:
                       np.reshape(data_dict['start_state'],
                                  [-1, self._observation_size])}
        )

        action = action_dist_mu + np.exp(action_dist_logstd) * \
            self._npr.randn(*action_dist_logstd.shape)
        return {'action': action, 'old_action_dist_mu': action_dist_mu,
                'old_action_dist_logstd': action_dist_logstd}

    def _generate_advantage(self, all_data_dict):

        for key in all_data_dict:
            data_dict = all_data_dict[key]
            # get the baseline function
            data_dict["value"] = self.value_pred(data_dict)
            # esitmate the advantages
            data_dict['advantage'] = np.zeros(data_dict['returns'].shape)
            start_id = 0
            #value = []

            print(np.mean(data_dict['returns']), flush=True)

            for i_episode_id in range(len(data_dict['episode_length'])):
                # the gamma discounted rollout value function
                current_length = data_dict['episode_length'][i_episode_id]
                end_id = start_id + current_length

                #value.append(data_dict['value'][start_id+1:end_id+1])
                for i_step in reversed(range(current_length)):
                    if i_step < current_length - 1:
                        delta = data_dict['returns'][i_step + start_id] \
                                + self.params.gamma * \
                                data_dict['value'][i_step + start_id + 1] \
                                - data_dict['value'][i_step + start_id]
                        data_dict['advantage'][i_step + start_id] = \
                            delta + self.params.gamma * self.params.gae_lam \
                            * data_dict['advantage'][i_step + start_id + 1]
                    else:
                        delta = data_dict['returns'][i_step + start_id] \
                                - data_dict['value'][i_step + start_id]
                        data_dict['advantage'][i_step + start_id] = delta
                start_id = end_id
            assert end_id == len(data_dict['rewards'])

            data_dict['value_target'] = \
                np.reshape(data_dict['advantage'], [-1, 1])
            data_dict['advantage'] -= data_dict['advantage'].mean()
            data_dict['advantage'] /= (data_dict['advantage'].std() + 1e-8)
            data_dict['advantage'] = np.reshape(data_dict['advantage'], [-1, 1])

    def _init_sparse_params(self):

        self._default_policy_masks = {}
        self._default_policy_grad = {}
        self._default_policy_params = {}
        self._default_policy_embed = {task_name:{} for task_name in self.task_names}

        self._default_value_masks = {}
        self._default_value_grad = {}
        self._default_value_params = {}
        self._default_value_embed = {task_name:{} for task_name in self.task_names}

        for x in self._input_ph['policy_embed']:
            shape = x.get_shape().as_list()
            dim_out = shape[1]
            n_task = len(self.task_names)

            full_rotation = generate_haar(dim_out)

            for i, task_name in enumerate(self.task_names):
                embed = full_rotation[i * n_task:(i+1)*n_task]
                self._default_policy_embed[x] = embed

        for x in self._input_ph['value_embed']:
            shape = x.get_shape().as_list()
            dim_out = shape[1]
            n_task = len(self.task_names)

            full_rotation = generate_haar(dim_out)

            for i, task_name in enumerate(self.task_names):
                embed = full_rotation[i * n_task:(i+1)*n_task]
                self._default_value_embed[x] = embed

        for task_name in self.task_names:
            p_masks = OrderedDict()
            p_grad = OrderedDict()
            p_params = OrderedDict()

            v_masks = OrderedDict()
            v_grad = OrderedDict()
            v_params = OrderedDict()
            for x in self._input_ph['policy_sparse_masks']:
                p_masks[x] = np.ones(x.get_shape().as_list())
                p_grad[x] = np.zeros(x.get_shape().as_list())
                p_params[x] = np.ones(x.get_shape().as_list())


            for y in self._input_ph['value_sparse_masks']:
                v_masks[y] = np.ones(y.get_shape().as_list())
                v_grad[y] = np.zeros(y.get_shape().as_list())
                v_params[y] = np.ones(y.get_shape().as_list())

            self._default_policy_masks[task_name] = p_masks
            self._default_policy_grad[task_name] = p_grad
            self._default_policy_params[task_name] = p_params

            self._default_value_masks[task_name] = v_masks
            self._default_value_grad[task_name] = v_grad
            self._default_value_params[task_name] = v_params

    def get_weights(self, task_name):
        weights = self._get_network_weights()
        for ix in range(len(self._tensor['policy_weights'])):
            var = self._tensor['policy_weights'][ix]
            name = var.name.replace(self._name_scope, '')
            assert name in weights
            weights[name] = weights[name] * \
                self._default_policy_masks[task_name][self._input_ph['policy_sparse_masks'][ix]]

            weights[name] = \
                self._default_policy_embed[task_name][self._input_ph['policy_embed'][ix]] @ \
                weights[name]

        for jx in range(len(self._tensor['value_weights'])):
            var = self._tensor['value_weights'][jx]
            name = var.name.replace(self._name_scope, '')
            assert name in weights
            weights[name] = weights[name] * \
                self._default_value_masks[task_name][self._input_ph['value_sparse_masks'][jx]]

            weights[name] = \
                self._default_value_embed[task_name][self._input_ph['value_embed'][ix]] @ \
                weights[name]

        return weights

    def set_weights(self, weight_dict):
        return self._set_network_weights(weight_dict)

    def value_pred(self, data_dict):
        return self._session.run(
            self._tensor['pred_value'],
            feed_dict={**self._default_policy_masks,
                       **self._default_value_masks,
                        self._input_ph['start_state']: data_dict['end_state']}
        )

    def get_sparse_weights(self):
        policy_weights = self._session.run(self._tensor['policy_weights'])
        value_weights = self._session.run(self._tensor['value_weights'])
        policy_b = self._session.run(self._tensor['policy_b'])
        value_b = self._session.run(self._tensor['value_b'])

        policy_return = []
        value_return = []
        for i in range(len(self._tensor['policy_weights'])):
            sparse_ix = self._default_policy_masks[self._input_ph['policy_sparse_masks'][i]]
            w = policy_weights[i] * sparse_ix
            policy_return.append((w, policy_b[i]))

        for i in range(len(self._tensor['value_weights'])):
            sparse_ix = self._default_value_masks[self._input_ph['value_sparse_masks'][i]]
            w = value_weights[i] * sparse_ix
            value_return.append((w, value_b[i]))

        return policy_return, value_return

    def update_sparse_weights(self, policy_gradients, value_gradients, task_name):

        for i in policy_gradients:
            gradient = policy_gradients[i]
            new_param = self._default_policy_params[task_name][i] + (np.abs(gradient) ** 2)
            new_mask = np.cast((self._default_policy_params[task_name][i] >= 0), np.int32)

            self._default_policy_grad[task_name][i] += gradient
            new_param, new_mask = self.reactivate_random(new_mask, new_param)
            self._default_policy_params[task_name][i] = new_param
            self._default_policy_masks[task_name][i] = new_mask

        for i in value_gradients:
            gradient = value_gradients[i]
            new_param = self._default_value_params[task_name][i] + (np.abs(gradient) ** 2)
            new_mask = np.cast((self._default_value_params[task_name][i] >= 0), np.int32)

            self._default_value_grad[task_name][i] += gradient
            new_param, new_mask = self.reactivate_random(new_mask, new_param)
            self._default_value_params[task_name][i] = new_param
            self._default_value_masks[task_name][i] = new_mask

        for task in self._default_policy_masks:
            if task == task_name:
                continue # skip

            other_policy_masks = self._default_policy_masks[task_name]
            other_policy_params = self._default_policy_params[task_name]
            other_policy_grad = self._default_policy_grad[task_name]
            for i in policy_gradients:
                gradient = policy_gradients[i]
                sign = 2 * np.cast((gradient * other_policy_grad[i]) > 0, np.int32) - 1
                new_param = other_policy_params[i] + sign * (np.abs(gradient) ** 2)
                new_mask = np.cast((new_param >= 0), np.int32)

                other_policy_grad[i] += gradient
                new_param, new_mask = self.reactivate_random(new_mask, new_param)
                other_policy_masks[i] = new_mask
                other_policy_params[i] = new_param


            other_value_masks = self._default_value_masks[task_name]
            other_value_params = self._default_value_params[task_name]
            other_value_grad = self._default_value_grad[task_name]

            for i in value_gradients:
                gradient = value_gradients[i]
                sign = 2 * np.cast((gradient * other_value_grad[i]) > 0, np.int32) - 1
                new_param = other_value_params[i] + sign * (np.abs(gradient) ** 2)
                new_mask = np.cast((new_param >= 0), np.int32)

                other_value_grad[i] += gradient
                new_param, new_mask = self.reactivate_random(new_mask, new_param)
                other_value_masks[i] = new_mask
                other_value_params[i] = new_param

    def reactivate_random(self, mask, params):
        if np.sum(mask) < np.size(mask) // self.params.num_tasks:
            to_replace = (np.size(mask) // self.params.num_tasks) - np.sum(mask)
            indices = np.array(mask.nonzero()).T
            num_ix = len(indices)
            chosen = self._npr.choice(np.arange(num_ix), to_replace, replace=False)
            mask[indices[chosen]] = 1
            params[indices[chosen]] = 1

        return mask, params
