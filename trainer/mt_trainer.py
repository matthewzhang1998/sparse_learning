import numpy as np
import multiprocessing
import os.path as osp
import tensorflow as tf
from util import parallel_util, whitening_util, \
    replay_buffer, logger, misc_util, init_path
from env import env_register


class Trainer(multiprocessing.Process):

    def __init__(self, params, network_type, task_queue, result_queue,
                 name_scope='trainer', task_names = ['none'], init_weights=None, path=None):
        multiprocessing.Process.__init__(self)
        self.params = params
        self._name_scope = name_scope
        self.task_names = task_names
        self.path = path

        # the base agent
        self._base_path = init_path.get_abs_base_dir()

        # used to save the checkpoint files
        self._iteration = 0
        self._best_reward = -np.inf
        self._timesteps_so_far = 0
        self._npr = np.random.RandomState(self.params.seed)
        self._task_queue = task_queue
        self._result_queue = result_queue
        self._network_type = network_type
        self.init_weights = init_weights

    def run(self):
        self._set_io_size()
        self._build_models()
        self._init_replay_buffer()
        self._init_whitening_stats()

        # load the model if needed
        if self.params.ckpt_name is not None:
            self._restore_all()

        # the main training process
        while True:
            next_task = self._task_queue.get()

            if next_task[0] is None or next_task[0] == parallel_util.END_SIGNAL:
                # kill the learner
                self._task_queue.task_done()
                break

            elif next_task[0] == parallel_util.START_SIGNAL:
                # get network weights
                self._task_queue.task_done()
                weights = {}
                for tn in self.task_names:
                    weights[tn] = self._get_weights(tn)

                self._result_queue.put(weights)

            elif next_task[0] == parallel_util.RESET_SIGNAL:
                self._task_queue.task_done()
                self._init_whitening_stats()
                self._timesteps_so_far = 0
                self._iteration = 0

            elif next_task[0] == parallel_util.FETCH_FINAL_WEIGHTS:
                weights = self.get_final_parameters()
                self._result_queue.put(weights)

            elif next_task[0] == parallel_util.SAVE_SIGNAL:
                _save_root = next_task[1]['net']
                _log_path = logger._get_path()

                _save_extension = _save_root + \
                                  "_{}_{}.ckpt".format(
                                      self._name_scope, self._timesteps_so_far
                                  )

                _save_dir = osp.join(_log_path, _save_extension)
                self._saver.save(self._session, _save_dir)

            elif next_task[0] == parallel_util.TRAINER_SET_WEIGHTS:
                weights = next_task[1]
                self._network.set_weights(weights)

            else:
                # training
                assert next_task[0] == parallel_util.TRAIN_SIGNAL
                stats = self._update_parameters(
                    next_task[1]['data'], next_task[1]['training_info']
                )
                self._task_queue.task_done()

                self._iteration += 1

                if self.params.separate_train:
                    pass

                else:
                    return_data = {}
                    for tn in self.task_names:
                        weights = self._get_weights(tn)

                        return_data[tn] = {
                            'network_weights': weights,
                            'stats': stats[tn],
                            'totalsteps': self._timesteps_so_far,
                            'iteration': self._iteration
                        }

                self._result_queue.put(return_data)

    def get_experiment_name(self):
        return 'ml'+ self.params.num_subtasks + '_' + self.params.exp_id

    def _build_session(self):
        config = tf.ConfigProto(device_count={'GPU': 0})  # only cpu version
        self._session = tf.Session(config=config)

    def _build_models(self):
        self._build_session()

        self._network = self._network_type(
            self.params, self._session, self._name_scope,
            self._observation_size, self._action_size,
            self._action_distribution, task_names = self.task_names,
            init_weights = self.init_weights, path=self.path
        )
        self._network.build_model()
        self._session.run(tf.global_variables_initializer())

        self._network.build_writer()
        self._saver = tf.train.Saver()

    def _set_io_size(self):
        self._observation_size, self._action_size, \
        self._action_distribution = \
            env_register.mt_io_information(self.params.num_subtasks)

    def _init_replay_buffer(self):
        if self._action_distribution is 'discrete':
            _action_size = None  # single discrete action only
        else:
            _action_size = self._action_size

        self._replay_buffer = replay_buffer.build_replay_buffer(
            self.params, self._observation_size, _action_size,
            save_reward=True
        )

    def _init_whitening_stats(self):
        self._whitening_stats = \
            whitening_util.init_whitening_stats(['state', 'diff_state'])

    def _update_whitening_stats(self, rollout_data,
                                key_list=['state', 'diff_state']):
        # collect the info
        rollout_data = [i_ep for data in rollout_data for i_ep in data]
        # rollout_data is list of list of dicts
        # -> list of dicts needed for update

        for key in key_list:
            whitening_util.update_whitening_stats(
                self._whitening_stats, rollout_data, key
            )

    def _preprocess_data(self, all_rollout_data):
        """ @brief:
                Process the data, collect the element of
                ['start_state', 'end_state', 'action', 'reward', 'return',
                 'ob', 'action_dist_mu', 'action_dist_logstd']
        """

        all_training_data = {}
        for tn in self.task_names:
            rollout_data = all_rollout_data[tn]
            # get the observations
            training_data = {}

            # get the returns (might be needed to train policy)
            for i_episode in rollout_data:
                i_episode["returns"] = \
                    misc_util.get_return(i_episode["rewards"],
                                          self.params.gamma)

            for key in self._network.required_keys:
                training_data[key] = np.concatenate(
                    [i_episode[key][:] for i_episode in rollout_data]
                )

            # record the length
            training_data['episode_length'] = \
                [len(i_episode['rewards']) for i_episode in rollout_data]

            # get the episodic reward
            for i_episode in rollout_data:
                i_episode['episodic_reward'] = sum(i_episode['rewards'])
            avg_reward = np.mean([i_episode['episodic_reward']
                                  for i_episode in rollout_data])
            logger.info('Mean reward: {}'.format(avg_reward))

            training_data['whitening_stats'] = self._whitening_stats
            training_data['avg_reward'] = avg_reward
            training_data['rollout_data'] = rollout_data

            all_training_data[tn] = training_data

        self._timesteps_so_far += len(training_data['action'])
        return all_training_data

    def _restore_all(self):
        # TODO
        pass

    def _save_all(self):
        # TODO
        pass

    def _get_weights(self, tn):
        weights = self._network.get_weights(tn)

        return weights

    def get_final_parameters(self):
        params = None
        for tn in self.task_names:
            t_params = self._network.get_weights(tn)
            if params is not None:
                for i,(x,y) in enumerate(zip(t_params, params)):
                    params[i] = x + y
            else:
                params = t_params

    def _update_parameters(self, rollout_data, *args):
        if self.params.separate_train:
            self._update_whitening_stats(rollout_data)
            training_data = self._preprocess_data(rollout_data)
            training_stats = {'avg_reward': training_data['avg_reward']}

            # train the policy
            stats_dictionary, data_dictionary = \
                self._network.train(
                    training_data,  self._replay_buffer
                )

            training_stats.update(stats_dictionary)

            self._replay_buffer.add_data(data_dictionary)

        else:
            training_stats, training_data = {}, {}

            for tn in self.task_names:
                self._update_whitening_stats(rollout_data[tn])

                training_data[tn] = self._preprocess_data(rollout_data[tn])
                training_stats[tn] = {'avg_reward': training_data[tn]['avg_reward']}

                # train the policy
            stats_dictionary, data_dictionary = \
                self._network.train(
                    training_data, self._replay_buffer
                )

            for tn in self.task_names:
                training_stats[tn].update(stats_dictionary[tn])

                self._replay_buffer.add_data(data_dictionary[tn])

        return training_stats