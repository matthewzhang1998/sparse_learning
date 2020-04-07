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
    def __init__(self, params, worker_proto, network_proto, subtask=None):
        self.params = params
        self._npr = np.random.RandomState(params.seed + 23333)
        self._observation_size, self._action_size, \
        self._action_distribution = \
            env_register.io_information(self.params.task)
        self._worker_type = worker_proto
        self._network_type = network_proto

        self.subtask = subtask

        # init the multiprocess actors
        self._task_queue = multiprocessing.JoinableQueue()
        self._result_queue = multiprocessing.Queue()
        self._init_workers()
        self._build_env()
        self._base_path = init_path.get_abs_base_dir()

        self._current_iteration = 0

    def set_weights(self, weights):
        for i_agent in range(self.params.num_workers):
            self._task_queue.put((parallel_util.AGENT_SET_WEIGHTS,
                                  weights))
        self._task_queue.join()

    def _init_workers(self):
        self._workers = []

        for worker in range(self.params.num_workers):
            self._workers.append(
                self._worker_type.Worker(
                    self.params, self._observation_size, self._action_size,
                    self._action_distribution,
                    self._network_type, self._task_queue, self._result_queue,
                    worker, subtask=self.subtask
                )
            )

        for worker in self._workers:
            worker.start()

    def _rollout_with_workers(self):
        self._current_iteration += 1
        rollout_data = []
        timesteps_needed = self.params.batch_size
        num_timesteps_received = 0

        while True:
            num_estimated_episode = int(
                timesteps_needed / self.params.episode_length
            )

            num_envs_per_worker = \
                num_estimated_episode / self.params.num_workers

            for _ in range(self.params.num_workers):
                self._task_queue.put((parallel_util.WORKER_RUNNING,
                                      num_envs_per_worker))

            self._task_queue.join()

            # collect the data
            for _ in range(num_estimated_episode):
                traj_episode = self._result_queue.get()
                rollout_data.append(traj_episode)
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
        self._env, self._env_info = env_register.make_env(
            self.params.task, self._npr.randint(0, 999999),
            self.params.episode_length,
            {'allow_monitor': self.params.monitor}
        )