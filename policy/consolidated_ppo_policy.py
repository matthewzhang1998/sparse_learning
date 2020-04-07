import sys
import os

import tensorflow as tf
import numpy as np
from policy import base_policy
from util import tf_util, network_util

TASKS = lambda x: 'task'+str(x)

class ConsolidatedPPOPolicy(base_policy.BasePolicy):
    def __init__(self, params, session, scope,
                 observation_size, action_size, *args, path=None, **kwargs):
        super(ConsolidatedPPOPolicy, self).__init__(params, session,
            scope, observation_size, action_size, path=path)

        self.required_keys = ['start_state', 'end_state', 'action', 'rewards', 'returns',
                                'old_action_dist_mu', 'old_action_dist_logstd']

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable

        self._trainable_var_list = \
            [[var for var in self._tensor['variable_list_{}'.format(TASKS(i))]]
             for i in range(self.params.num_subtasks)]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            [var_list + self._whitening_variable
             for var_list in self._trainable_var_list]


        self._all_trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._set_network_weights = tf_util.set_network_weights(
            self._session, self._all_trainable_var_list, self._name_scope
        )

        self._get_network_weights = [tf_util.get_network_weights(
            self._session, self._network_var_list[i],
            self._name_scope, strip_end = '_{}'.format(TASKS(i))
        ) for i in range(self.params.num_subtasks)]

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
            self._MLP = network_util.SparseMultitaskMLP(
                dims=network_shape, scope='policy_mlp', train=True,
                activation_type=act_type, normalizer_type=norm_type,
                init_data=init_data, linear_last_layer=True,
                num_tasks=self.params.num_subtasks
            )

            self._tensor['policy_b'] = self._MLP._b
            output = self._MLP(self._tensor['net_input'])
            for i in range(self.params.num_subtasks):
                self._tensor['policy_weights_{}'.format(TASKS(i))] = self._MLP._w[i]
                self._tensor['policy_masks_{}'.format(TASKS(i))] = self._MLP._sparse_mask[i]

                # the output policy of the network
                self._tensor['action_dist_mu_{}'.format(TASKS(i))] = output[i]

            # use uniform logstd
            self._tensor['action_logstd'] = tf.Variable(
                (0 * self._npr.randn(1, self._action_size)).astype(np.float32),
                name="action_logstd", trainable=True
            )

            self._tensor['action_dist_logstd'] = tf.tile(
                self._tensor['action_logstd'],
                [tf.shape(self._tensor['action_dist_mu_task0'])[0], 1]
            )
            # make sure the size is matched to [batch, num_action]
            # fetch all the trainable variables

            self.build_loss()

    def build_loss(self):
        self._update_operator = {}
        self._build_value_network_and_loss()
        self._set_var_list()
        self._build_ppo_loss_preprocess()
        self._build_ppo_loss()

    def build_writer(self):

        self.Writer = tf.summary.FileWriter(
            os.path.join(self.path, self._name_scope), self._session.graph)

        self.Summary = {}

        for i in range(self.params.num_subtasks):
            for j in range(len(self._tensor['policy_weights_{}'.format(TASKS(0))])):
                mask = tf.expand_dims(tf.expand_dims(
                    self._tensor['policy_masks_{}'.format(TASKS(i))][j], 0), 3)
                weight = tf.expand_dims(tf.expand_dims(
                    tf.sigmoid(self._tensor['policy_weights_{}'.format(TASKS(i))][j]), 0), 3)

                self.Summary['Weights_{}_{}'.format(TASKS(i), j)] = tf.summary.image(
                    'Weights_{}_{}'.format(TASKS(i), j), weight)

                self.Summary['Masks_{}_{}'.format(TASKS(i), j)] = tf.summary.image(
                    'Masks_{}_{}'.format(TASKS(i), j), mask)

        summary = self._session.run(self.Summary)
        for key in summary:
            self.Writer.add_summary(summary[key], self._timesteps_so_far)
            #self._init_sparse_params()

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

        if self.params.use_subtask_value:
            self._value_MLP = network_util.SparseMultitaskMLP(
                dims=network_shape, scope='value_mlp', train=True,
                activation_type=act_type, normalizer_type=norm_type,
                init_data=init_data, linear_last_layer=True,
                num_tasks=self.params.num_subtasks
            )

            for i in range(self.params.num_subtasks):
                self._tensor['value_weights_{}'.format(TASKS(i))] = self._value_MLP._w[i]
                self._tensor['value_masks_{}'.format(TASKS(i))] = self._value_MLP._sparse_mask[i]

            self._tensor['value_b'] = self._value_MLP._b

            output = self._value_MLP(self._tensor['net_input'])

            self.value_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params.value_lr,
                beta1=0.5, beta2=0.99, epsilon=1e-4
            )

            for i in range(self.params.num_subtasks):
                self._tensor['pred_value_{}'.format(TASKS(i))] = output[i]

                self._update_operator['vf_loss_{}'.format(TASKS(i))] = .5 * tf.reduce_mean(
                    tf.square(
                        self._tensor['pred_value_{}'.format(TASKS(i))] - self._input_ph['value_target']
                    )
                )

                self._update_operator['sparse_value_correlation_loss_{}'] = \
                    tf_util.correlation_loss(self._tensor['value_masks_{}'.format(TASKS(i))],
                     [self._tensor['value_masks_{}'.format(TASKS(j))] \
                      for j in range(self.params.num_subtasks) if j != i])

                self._update_operator['sparse_value_mask_loss_{}'] = \
                    tf_util.l2_loss(self._tensor['value_masks_{}'.format(TASKS(i))],
                        apply_sigmoid=True)

                self._update_operator['vf_loss_{}'.format(TASKS(i))] += \
                    self._update_operator['sparse_value_correlation_loss_{}'] * \
                    self.params.correlation_coefficient + \
                    self._update_operator['sparse_value_mask_loss_{}'] * \
                    self.params.mask_penalty

                self._update_operator['vf_update_op_{}'.format(TASKS(i))] = \
                    self.value_optimizer.minimize(self._update_operator['vf_loss_{}'.format(TASKS(i))])

                self._tensor['variable_list_{}'.format(TASKS(i))] = [
                    *self._tensor['policy_weights_{}'.format(TASKS(i))],
                    *self._tensor['policy_b'],
                    *self._tensor['value_weights_{}'.format(TASKS(i))],
                    *self._tensor['value_b'],
                    self._tensor['action_logstd']
                ]

        else:
            self._value_MLP = network_util.MLP(
                dims=network_shape, scope='value_mlp', train=True,
                activation_type=act_type, normalizer_type=norm_type,
                init_data=init_data, linear_last_layer=True,
            )

            self._tensor['value_weights'] = self._value_MLP._w
            self._tensor['value_b'] = self._value_MLP._b

            self._tensor['pred_value'] = self._value_MLP(self._tensor['net_input'])

            self.value_optimizer = tf.train.AdamOptimizer(
                learning_rate=self.params.value_lr,
                beta1=0.5, beta2=0.99, epsilon=1e-4
            )
            self._update_operator['vf_loss'] = .5 * tf.reduce_mean(
                tf.square(
                    self._tensor['pred_value'] - self._input_ph['value_target']
                )
            )

            self._update_operator['vf_update_op'] = \
                self.value_optimizer.minimize(self._update_operator['vf_loss'])

            for i in range(self.params.num_subtasks):
                self._tensor['variable_list_{}'.format(TASKS(i))] = [
                    *self._tensor['policy_weights_{}'.format(TASKS(i))],
                    *self._tensor['policy_b'],
                    *self._tensor['value_weights'],
                    *self._tensor['value_b'],
                    self._tensor['action_logstd']
                ]

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

        self._tensor['log_oldp_n'] = tf_util.gauss_log_prob(
            self._input_ph['old_action_dist_mu'],
            self._input_ph['old_action_dist_logstd'],
            self._input_ph['action']
        )

        # the kl and ent of the policy
        for i in range(self.params.num_subtasks):
            self._tensor['log_p_n_{}'.format(TASKS(i))] = tf_util.gauss_log_prob(
                self._tensor['action_dist_mu_{}'.format(TASKS(i))],
                self._tensor['action_dist_logstd'],
                self._input_ph['action']
            )

            self._tensor['ratio_{}'.format(TASKS(i))] = \
                tf.exp(self._tensor['log_p_n_{}'.format(TASKS(i))] - self._tensor['log_oldp_n'])

            self._tensor['ratio_clipped_{}'.format(TASKS(i))] = tf.clip_by_value(
                self._tensor['ratio_{}'.format(TASKS(i))], 1. - self._ppo_clip, 1. + self._ppo_clip
            )

            # the kl divergence between the old and new action
            self._tensor['kl_{}'.format(TASKS(i))] = tf_util.gauss_KL(
                self._input_ph['old_action_dist_mu'],
                self._input_ph['old_action_dist_logstd'],
                self._tensor['action_dist_mu_{}'.format(TASKS(i))],
                self._tensor['action_dist_logstd']
            ) / self._input_ph['batch_size']

            self._tensor['ent_{}'.format(TASKS(i))] = tf_util.gauss_ent(
                self._tensor['action_dist_mu_{}'.format(TASKS(i))],
                self._tensor['action_dist_logstd']
            ) / self._input_ph['batch_size']

    def _build_ppo_loss(self):
        self.policy_optimizer =  tf.train.AdamOptimizer(
                learning_rate=self._input_ph['lr'],
            )

        for i in range(self.params.num_subtasks):
            self._update_operator['pol_loss_unclipped_{}'.format(TASKS(i))] = \
                -self._tensor['ratio_{}'.format(TASKS(i))] * \
                tf.reshape(self._input_ph['advantage'], [-1])

            self._update_operator['pol_loss_clipped_{}'.format(TASKS(i))] = \
                -self._tensor['ratio_clipped_{}'.format(TASKS(i))] * \
                tf.reshape(self._input_ph['advantage'], [-1])

            self._update_operator['surr_loss_{}'.format(TASKS(i))] = tf.reduce_mean(
                tf.maximum(self._update_operator['pol_loss_unclipped_{}'.format(TASKS(i))],
                           self._update_operator['pol_loss_clipped_{}'.format(TASKS(i))])
            )

            self._update_operator['loss_{}'.format(TASKS(i))] = \
                self._update_operator['surr_loss_{}'.format(TASKS(i))]

        # if self.params.use_kl_penalty:
        #     self._update_operator['kl_pen'] = \
        #         self._input_ph['kl_lambda'] * \
        #         self._tensor['kl']
        #     self._update_operator['kl_loss'] = self._kl_eta * \
        #         tf.square(tf.maximum(0., self._tensor['kl'] -
        #                              2. * self.params.target_kl))
        #     self._update_operator['loss'] += \
        #         self._update_operator['kl_pen'] + \
        #         self._update_operator['kl_loss']

            if self.params.use_weight_decay:
                self._update_operator['weight_decay_loss_{}'] = \
                    tf_util.l2_loss(self._tensor['variable_list_{}'.format(TASKS(i))])
                self._update_operator['loss_{}'.format(TASKS(i))] += \
                    self._update_operator['weight_decay_loss_{}'.format(TASKS(i))] * \
                    self.params.weight_decay_coefficient

            self._update_operator['sparse_correlation_loss_{}'.format(TASKS(i))] = \
                tf_util.correlation_loss(self._tensor['policy_masks_{}'.format(TASKS(i))],
                    [self._tensor['policy_masks_{}'.format(TASKS(j))] \
                    for j in range(self.params.num_subtasks) if j != i], apply_sigmoid=True)

            self._update_operator['sparse_mask_loss_{}'.format(TASKS(i))] = \
                tf_util.l2_loss(self._tensor['policy_masks_{}'.format(TASKS(i))],
                apply_sigmoid=True)

            self._update_operator['surr_loss_{}'.format(TASKS(i))] += \
                self._update_operator['sparse_correlation_loss_{}'.format(TASKS(i))] * \
                self.params.correlation_coefficient + \
                self._update_operator['sparse_mask_loss_{}'.format(TASKS(i))] * \
                self.params.mask_penalty

            self._update_operator['update_op_{}'.format(TASKS(i))] = \
                self.policy_optimizer.minimize(self._update_operator['surr_loss_{}'.format(TASKS(i))])


        # # the actual parameter values
        # self._update_operator['get_flat_param'] = \
        #     tf_util.GetFlat(self._session, self._trainable_var_list)
        # # call this to set parameter values
        # self._update_operator['set_from_flat_param'] = \
        #     tf_util.SetFromFlat(self._session, self._trainable_var_list)

    def train(self, joint_data_dict, replay_buffer, training_info={}):
        stats = [{'surr_loss': [], 'entropy': [], 'kl': [], 'vf_loss': []}
                 for _ in range(self.params.num_subtasks)]

        update_kl_mean = -.1

        self._iters_so_far += 1

        self._timesteps_so_far += len(joint_data_dict[0]['start_state'])

        for i in range(self.params.num_subtasks):
            self._generate_advantage(joint_data_dict[i], i)

        for epoch in range(max(self.params.policy_epochs,
                               self.params.value_epochs)):
            for i in range(len(joint_data_dict)):
                data_dict = joint_data_dict[i]

                total_batch_len = len(data_dict['start_state'])
                total_batch_inds = np.arange(total_batch_len)
                self._npr.shuffle(total_batch_inds)
                minibatch_size = total_batch_len // self.params.num_minibatches
                kl_stopping = False

                for start in range(self.params.num_minibatches):
                    start = start * minibatch_size
                    end = min(start + minibatch_size, total_batch_len)
                    batch_inds = total_batch_inds[start:end]
                    feed_dict={
                        self._input_ph[key]: data_dict[key][batch_inds]
                        for key in ['start_state', 'action', 'advantage',
                                    'old_action_dist_mu', 'old_action_dist_logstd']
                    }

                    feed_dict[self._input_ph['batch_size']] = \
                        np.array(float(end - start))

                    feed_dict[self._input_ph['lr']] = self._current_lr

                    if epoch < self.params.policy_epochs:
                        loss, kl, _ = self._session.run(
                            [self._update_operator['surr_loss_{}'.format(TASKS(i))],
                             self._tensor['kl_{}'.format(TASKS(i))],
                             self._update_operator['update_op_{}'.format(TASKS(i))]],
                            feed_dict
                        )
                        # kl = self._session.run([self._tensor['kl']], feed_dict)
                        stats[i]['surr_loss'].append(loss)
                        stats[i]['kl'].append(kl)

                    if epoch < self.params.value_epochs:
                        # train the baseline function
                        feed_dict[self._input_ph['old_values']] = \
                            data_dict['value'][batch_inds]
                        feed_dict[self._input_ph['value_target']] = \
                            data_dict['value_target'][batch_inds]

                        if self.params.use_subtask_value:
                            vf_loss, _ = self._session.run(
                                [self._update_operator['vf_loss_{}'.format(TASKS(i))],
                                 self._update_operator['vf_update_op_{}'.format(TASKS(i))]],
                                feed_dict=feed_dict
                            )

                        else:
                            vf_loss, _ = self._session.run(
                                [self._update_operator['vf_loss'],
                                 self._update_operator['vf_update_op']],
                                feed_dict=feed_dict
                            )

                        stats[i]['vf_loss'].append(vf_loss)

                    if self.params.use_kl_penalty:
                        if self.params.num_minibatches > 1:
                            raise RuntimeError('KL penalty not available \
                                               with minibatches')
                        else:
                            if np.mean(stats[i]['kl']) > 4 * self.params.target_kl:
                                kl_stopping = True

                # break condition
                if kl_stopping:
                    update_kl_mean = np.mean(stats[i]['kl'])
                    break

        # run final feed_dict with whole batch

        kl_total = 0
        for i in range(len(joint_data_dict)):
            feed_dict={
                self._input_ph[key]: joint_data_dict[i][key]
                for key in ['start_state', 'action', 'advantage',
                            'old_action_dist_mu', 'old_action_dist_logstd']
            }

            feed_dict[self._input_ph['batch_size']] = \
                np.array(float(total_batch_len))

            kl_total += self._session.run([self._tensor['kl_{}'.format(TASKS(i))]], feed_dict)[0]

        summary = self._session.run(self.Summary)
        for key in summary:
            self.Writer.add_summary(summary[key], self._timesteps_so_far)

        self._update_adaptive_parameters(update_kl_mean, kl_total)
        if self.params.use_kl_penalty:
            self._current_kl_lambda = self._current_kl_lambda
        # update the whitening variables

        self._set_whitening_var(joint_data_dict[0]['whitening_stats']) # should be same for both tasks

        for i in range(self.params.num_subtasks):
            for key in stats[i]:
                stats[i][key] = np.mean(stats[i][key])

            stats[i]['advantage'] = np.mean(data_dict['advantage'])
            stats[i]['mean_value'] = np.mean(data_dict['value'])

        return stats, joint_data_dict

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

        if self._iters_so_far % self.params.sparsification_iter == 0:
            pass
            #self.update_sparsify(self._iters_so_far)

    def act(self, data_dict, *params):
        action_dist_mu, action_dist_logstd = self._session.run(
            [self._tensor['action_dist_mu'], self._tensor['action_dist_logstd']],
            feed_dict={**self._default_policy_masks,
                       **self._default_value_masks, self._input_ph['start_state']:
                       np.reshape(data_dict['start_state'],
                                  [-1, self._observation_size])}
        )

        action = action_dist_mu + np.exp(action_dist_logstd) * \
            self._npr.randn(*action_dist_logstd.shape)
        return {'action': action, 'old_action_dist_mu': action_dist_mu,
                'old_action_dist_logstd': action_dist_logstd}

    def _generate_advantage(self, data_dict, index=0):
        # get the baseline function
        data_dict["value"] = self.value_pred(data_dict, index)
        # esitmate the advantages
        data_dict['advantage'] = np.zeros(data_dict['returns'].shape)
        start_id = 0
        #value = []

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
            np.reshape(data_dict['advantage'], [-1, 1]) #+ np.reshape(data_dict['value'], [-1, 1])
        # from util.common.fpdb import fpdb; fpdb().set_trace()
        # standardized advantage function
        data_dict['advantage'] -= data_dict['advantage'].mean()
        data_dict['advantage'] /= (data_dict['advantage'].std() + 1e-8)
        data_dict['advantage'] = np.reshape(data_dict['advantage'], [-1, 1])

    def get_weights(self):
        weights = [self._get_network_weights[i]() for i in range(self.params.num_subtasks)]
        return weights

    def set_weights(self, weight_dict):
        assert False # trainer object do not call
        return self._set_network_weights(weight_dict)

    def value_pred(self, data_dict, i=None):

        if self.params.use_subtask_value:
            return self._session.run(
                self._tensor['pred_value_{}'.format(TASKS(i))],
                feed_dict={self._input_ph['start_state']: data_dict['end_state']}
            )

        else:
            return self._session.run(
                self._tensor['pred_value'.format(TASKS(i))],
                feed_dict={self._input_ph['start_state']: data_dict['end_state']}
            )


    def get_sparse_weights(self):
        policy_weights = self._session.run([self._tensor['policy_weights_{}'.format(TASKS(i))]
            for i in range(self.params.num_subtasks)])

        if self.params.use_subtask_value:
            value_weights = self._session.run([self._tensor['value_weights_{}'.format(TASKS(i))]
                for i in range(self.params.num_subtasks)])

        else:
            value_weights = self._session.run([self._tensor['value_weights']])[0]

        policy_b = self._session.run(self._tensor['policy_b'])
        value_b = self._session.run(self._tensor['value_b'])

        policy_return = [[]] * self.params.num_subtasks
        value_return = [[]] * self.params.num_subtasks
        for task in range(self.params.num_subtasks):
            for i in range(len(self._tensor['policy_weights_{}'.format(TASKS(0))])):
                w = policy_weights[task][i]
                policy_return[task].append((w, policy_b[i]))

            if self.params.use_subtask_value:
                for i in range(len(self._tensor['value_weights_{}'.format(TASKS(0))])):
                    w = value_weights[task][i]
                    value_return[task].append((w, value_b[i]))
            else:
                for i in range(len(self._tensor['value_weights'])):
                    w = value_weights[i]/2
                    value_return[task].append((w, value_b[i]))

        return policy_return, value_return
