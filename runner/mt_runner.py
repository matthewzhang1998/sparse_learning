#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 11 19:04:26 2018
@author: matthewszhang
"""
'''
CREDIT TO TINGWU WANG FOR THIS CODE
'''

import multiprocessing
from util import init_path
from util import parallel_util
from util import logger
from env import env_register
import numpy as np


class Sampler(object):
    def __init__(self, params, worker_proto, network_proto, task_names=None):
        self.params = params
        self._npr = np.random.RandomState(params.seed + 23333)
        self._observation_size, self._action_size, \
        self._action_distribution = \
            env_register.mt_io_information(self.params.num_subtasks)
        self._worker_type = worker_proto
        self._network_type = network_proto

        self.task_names = task_names

        # init the multiprocess actors
        self._task_queue = multiprocessing.JoinableQueue()
        self._result_queue = multiprocessing.Queue()
        self._init_workers()
        self._build_env()
        self._base_path = init_path.get_abs_base_dir()

        self._current_iteration = 0

    def set_weights_multi(self, weights):
        for i_agent in range(self.params.num_workers):
            self._task_queue.put((parallel_util.AGENT_SET_WEIGHTS_MULTI,
                                  weights))
        self._task_queue.join()

    def set_weights(self, weights):
        for i_agent in range(self.params.num_workers):
            self._task_queue.put((parallel_util.AGENT_SET_WEIGHTS,
                                  weights))
        self._task_queue.join()

    def _init_workers(self):
        self._workers = []

        for i in self.task_names:
            self._workers.append(
                self._worker_type.Worker(
                    self.params, self._observation_size, self._action_size,
                    self._action_distribution,
                    self._network_type, self._task_queue, self._result_queue,
                    i, task_name=i
                )
            )

        for worker in self._workers:
            worker.start()

    def _rollout_with_workers(self):
        self._current_iteration += 1
        rollout_data = {tn: [] for tn in self.task_names}
        timesteps_needed = self.params.batch_size
        num_timesteps_received = 0

        while True:
            num_estimated_episode = int(
                timesteps_needed / self.params.episode_length
            )

            num_envs_per_worker = \
                num_estimated_episode / self.params.num_workers

            for i in self.task_names:
                self._task_queue.put((parallel_util.WORKER_RUNNING_MT,
                                      num_envs_per_worker))

            self._task_queue.join()

            # collect the data
            for _ in range(num_estimated_episode):
                traj_episode = self._result_queue.get()

                rollout_data[traj_episode['task name']].append(traj_episode)
                num_timesteps_received += len(traj_episode['rewards'])

            logger.info('{} timesteps from {} episodes collected'.format(
                num_timesteps_received, len(rollout_data))
            )

            if num_timesteps_received >= timesteps_needed:
                break

        return {'data': rollout_data}

    def render(self, it, save_loc):
        self._task_queue.put((parallel_util.AGENT_RENDER, {'it': it, 'save_loc': save_loc}))
        self._task_queue.join()

    def end(self):
        for i_agent in range(self.params.num_workers):
            self._task_queue.put((parallel_util.END_ROLLOUT_SIGNAL, None))

    def _build_env(self):
        self._env, self._env_info = \
            env_register.make_mt_env(1, self.params.num_subtasks)