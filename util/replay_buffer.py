# -----------------------------------------------------------------------------
#   @brief: save the true datapoints into a buffer
#   @author: Tingwu Wang
# -----------------------------------------------------------------------------
import numpy as np


class replay_buffer(object):

    def __init__(self, use_buffer, buffer_size, rand_seed,
                 observation_size, action_size, save_reward=False):

        self._use_buffer = use_buffer
        self._buffer_size = buffer_size
        self._npr = np.random.RandomState(rand_seed)

        if not self._use_buffer:
            self._buffer_size = 0

        self._observation_size = observation_size
        self._action_size = action_size

        reward_data_size = self._buffer_size
        self._data = {
            'start_state': np.zeros(
                [self._buffer_size, self._observation_size],
                dtype=np.float16
            ),

            'end_state': np.zeros(
                [self._buffer_size, self._observation_size],
                dtype=np.float16
            ),

            'rewards': np.zeros([reward_data_size], dtype=np.float16)
        }

        if action_size is None:
            self._data['action'] = np.zeros(
                [self._buffer_size], dtype=np.int16
            )

        else:
            self._data['action'] = np.zeros(
                [self._buffer_size, self._action_size], dtype=np.int16
            )

        self._data_key = [key for key in self._data
                          if len(self._data[key]) > 0]

        self._current_id = 0
        self._occupied_size = 0

    def add_data(self, new_data):
        if self._buffer_size == 0:
            return

        num_new_data = len(new_data['start_state'])

        if num_new_data + self._current_id > self._buffer_size:
            num_after_full = \
                num_new_data + self._current_id - self._buffer_size
            for key in self._data_key:
                # filling the tail part
                self._data[key][self._current_id: self._buffer_size] = \
                    new_data[key][0: self._buffer_size - self._current_id]

                # filling the head part
                self._data[key][0: num_after_full] = \
                    new_data[key][self._buffer_size - self._current_id:]

        else:
            for key in self._data_key:
                self._data[key][self._current_id:
                                self._current_id + num_new_data] = \
                    new_data[key]

        self._current_id = \
            (self._current_id + num_new_data) % self._buffer_size
        self._occupied_size = \
            min(self._buffer_size, self._occupied_size + num_new_data)

    def get_data(self, batch_size):

        # the data from old data
        sample_id = self._npr.randint(0, self._occupied_size, batch_size)
        return {key: self._data[key][sample_id] for key in self._data_key}

    def get_current_size(self):
        return self._occupied_size

    def get_all_data(self):
        return {key: self._data[key][:self._occupied_size]
                for key in self._data_key}


class prioritized_replay_buffer(replay_buffer):
    def __init__(self, priority_alpha, *args, **kwargs):
        super(prioritized_replay_buffer, self).__init__(*args, **kwargs)
        self._alpha = priority_alpha

        # priority key
        self._data['priority'] = np.zeros((self._buffer_size,),
                                          dtype=np.float16)

        self._running_mean_priority = 0
        self._running_mean_size = 0

    def add_data(self, new_data):
        if self._buffer_size == 0:
            return

        num_new_data = len(new_data['start_state'])

        new_data['priority'] = np.mean(
            np.abs(new_data['rewards']) * \
            np.exp(-new_data['log_oldp_n'][:, :-1]),
            axis=-1
        )

        self._running_mean_size += num_new_data
        self._running_mean_priority += np.sum(new_data['priority']) / \
                                       self._running_mean_size

        new_data['priority'] /= self._running_mean_priority

        if num_new_data + self._occupied_size > self._buffer_size:

            # compute priorities for all data
            joint_priority = np.concatenate(
                (new_data['priority'],
                 self._data['priority'][:self._occupied_size]), axis=0
            )

            select_probs = np.power(joint_priority, self._alpha) / \
                           np.sum(np.power(joint_priority, self._alpha))

            select_inds = np.random.choice(
                num_new_data + self._occupied_size,
                size=self._buffer_size,
                p=select_probs
            )

            original_inds = np.where(select_inds < self._occupied_size)
            new_inds = np.where(select_inds >= self._occupied_size) - \
                       self._occupied_size

            for key in self._data_key:
                self._data[key] = np.concatenate(
                    self._data[key][original_inds],
                    new_data[key][new_inds], axis=0
                )
        else:
            for key in self._data_key:
                self._data[key][self._occupied_size:
                                self._occupied_size + num_new_data] = \
                    new_data[key]

        self._occupied_size = \
            min(self._occupied_size + num_new_data, self._buffer_size)

    def get_data(self, batch_size):
        select_probs = np.power(
            self._data['priority'][:self._occupied_size], axis=0
        ) / \
                       np.sum(
                           np.power(
                               self._data['priority'][:self._occupied_size], axis=0
                           )
                       )

        select_inds = np.random.choice(
            self._occupied_size,
            size=batch_size,
            p=select_probs
        )

        return {key: self._data[key][select_inds] for key in self._data_key}


class prioritized_recurrent_replay_buffer(prioritized_replay_buffer):
    def __init__(self, episode_len, *args, **kwargs):
        super(prioritized_recurrent_replay_buffer, self).__init__(
            *args, **kwargs
        )
        self._episode_len = episode_len

        self._buffer_episodes = int(self._buffer_size / episode_len)

        for key in self._data:
            self._data[key] = np.reshape(self._data[key],
                                         (self._buffer_episodes, episode_len,
                                          *self._data[key].shape[1:])
                                         )

        # overwrite priorities to be per episode
        self._data['priority'] = np.zeros((self._buffer_episodes,),
                                          dtype=np.float16)

    def add_data(self, new_data):
        if self._buffer_size == 0:
            return

        num_new_data = int(len(new_data['start_state']) / self._episode_len)

        new_priorities = np.mean(
            np.abs(new_data['advantage']) * \
            np.exp(-new_data['log_oldp_n'][:, :-1]),
            axis=-1
        )

        new_priorities = np.mean(
            np.reshape(
                new_priorities, [num_new_data, self._episode_len]
            ), axis=-1
        )

        new_data['priority'] = new_priorities

        self._running_mean_size += num_new_data
        self._running_mean_priority += np.sum(new_data['priority']) / \
                                       self._running_mean_size

        new_data['priority'] /= self._running_mean_priority

        if num_new_data + self._occupied_size > self._buffer_episodes:

            # compute priorities for all data
            joint_priority = np.concatenate(
                (new_data['priority'],
                 self._data['priority'][:self._occupied_size]), axis=0
            )

            select_probs = np.power(joint_priority, self._alpha) / \
                           np.sum(np.power(joint_priority, self._alpha))

            select_inds = np.random.choice(
                num_new_data + self._occupied_size,
                size=self._buffer_episodes,
                p=select_probs
            )

            original_inds = \
                select_inds[np.where(select_inds < self._occupied_size)]

            new_inds = \
                select_inds[np.where(select_inds >
                                     (self._occupied_size - 1))] - \
                self._occupied_size

            for key in self._data_key:
                if key is not 'priority':
                    new_data_by_episode = np.reshape(new_data[key],
                                                     (num_new_data, self._episode_len,
                                                      *np.shape(new_data[key])[1:])
                                                     )[new_inds]

                    self._data[key] = np.concatenate((
                        self._data[key][original_inds],
                        new_data_by_episode), axis=0
                    )

                else:
                    self._data[key] = np.concatenate(
                        (self._data[key][original_inds],
                         new_data[key][new_inds]), axis=0
                    )
        else:
            for key in self._data_key:
                if key is not 'priority':
                    new_data_by_episode = np.reshape(new_data[key],
                                                     ([num_new_data, self._episode_len,
                                                       *new_data[key].shape[1:]])
                                                     )

                    self._data[key][self._occupied_size: \
                                    self._occupied_size + num_new_data] = \
                        new_data_by_episode

                else:
                    self._data[key] = np.concatenate(
                        (self._data[key][original_inds],
                         new_data[key][new_inds]), axis=0
                    )

        self._occupied_size = \
            min(self._occupied_size + num_new_data, self._buffer_episodes)

    def get_data(self, batch_size):
        if batch_size > (self._occupied_size * self._episode_len):
            return None

        episode_size = int(batch_size / self._episode_len)
        select_probs = np.power(
            self._data['priority'][:self._occupied_size], self._alpha
        ) / \
                       np.sum(
                           np.power(
                               self._data['priority'][:self._occupied_size], self._alpha
                           )
                       )

        select_inds = np.random.choice(
            self._occupied_size,
            size=episode_size,
            p=select_probs
        )

        return_dict = \
            {key: self._data[key][select_inds] for key in self._data_key}

        # pad each of the returned arrays appropriately
        return_dict['start_state'] = np.append(return_dict['start_state'],
                                               return_dict['end_state'][:, -1, :][:, np.newaxis], axis=1
                                               )
        #        return_dict['initial_goals'] = np.append(
        #            return_dict['initial_goals'],
        #            return_dict['initial_goals'][:,0][:, np.newaxis], axis=1
        #        )
        for key in return_dict:
            return_dict[key] = np.reshape(return_dict[key],
                                          (-1, *return_dict[key].shape[2:])
                                          )

        return return_dict


def build_replay_buffer(args, observation_size, action_size,
                        save_reward=True):

    return replay_buffer(args.use_replay_buffer, args.replay_buffer_size,
        args.seed, observation_size, action_size, save_reward)



class DummyEpisodicBuffer(object):
    def __init__(self, double_dict_list, seed=0):
        assert double_dict_list[0][0] is not None

        buffer = {}

        for key in double_dict_list[0][0]:
            buffer[key] = []

        for key in buffer:
            for traj in double_dict_list:
                traj_key = []
                for timestep in traj:
                    traj_key.append(timestep[key])
                traj_key = np.array(traj_key)
                buffer[key].append(traj_key)
            buffer[key] = np.array(buffer[key])

        self.buffer = buffer
        self.num_trajectories = len(double_dict_list)
        self._npr = np.random.RandomState(seed)

    def sample(self, num_trajectories):
        assert num_trajectories <= self.num_trajectories

        sample_idx = self._npr.randint(
            0, self.num_trajectories, num_trajectories
        )

        data_dict = {}
        for key in self.buffer:
            data_dict[key] = self.buffer[key][sample_idx]
            data_dict[key] = data_dict[key].reshape(
                (data_dict[key].shape[0] * data_dict[key].shape[1],
                 data_dict[key].shape[2:])
            )

        return data_dict


def make_dummy_buffer(double_list_dict, seed=0):
    '''
    :param double_dict_list: list of trajectories, each of which is
        list of per-timestep dictionaries
    :return: unprioritized buffer organized by episode, but sampling as 1d array
    '''

    return DummyEpisodicBuffer(double_list_dict, seed)