#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 16:27:44 2018
@author: matthewszhang
"""
import numpy as np
from collections import defaultdict
import glob
import pickle
import os
import os.path as osp


def play_episode_with_env(envs, policy, control_info = {}):
    # init the variables
    return_infos = defaultdict(list)
    # 1 timestep copy
    feed_infos = defaultdict(list)
    control_info['step_index'] = 0
    control_info['reset'] = True

    # start the env (reset the environment)
    for env in envs:
        ob, _, _, _ = env.reset()
        feed_infos['start_state'].append(ob)

    return_infos['start_state'].append(
        np.array(feed_infos['start_state'])
    )

    while True:
        # generate the policy
        action_signal = policy(feed_infos, control_info)
        control_info['step_index'] += 1
        control_info['reset'] = False

        # only use initial state on first state
        control_info['use_default_states'] = False

        # reset feed_infos after feed
        feed_infos = defaultdict(list)

        # record the stats
        for key in action_signal:
            feed_infos[key] = np.array(action_signal[key])

        # take the action

        for i in range(len(envs)):
            ob, reward, done, _ = envs[i].step(action_signal['action'][i])
            if not done:
                feed_infos['start_state'].append(ob)
            feed_infos['end_state'].append(ob)
            feed_infos['rewards'].append(reward)

        for key in feed_infos:
            return_infos[key].append(np.array(feed_infos[key]))

        if done:
            break

    for key in return_infos:
        return_infos[key] = np.swapaxes(np.asarray(return_infos[key]), 0, 1)

    all_episode_infos = []
    for env in range(len(envs)):
        episode_info = {}
        for key in return_infos:
            episode_info[key] = return_infos[key][env]

        all_episode_infos.append(episode_info)

    return all_episode_infos


def load_environments(load_path, num_envs,
                      task_name, maximum_length=100, seed=0):
    from . import env_register

    root_dir = osp.join(os.getcwd(), '..', load_path)

    all_environments = [
        file for file in os.listdir(root_dir)
        if osp.isfile(osp.join(root_dir, file))
    ]

    environment_cache = []
    for i in range(num_envs):
        env, _ = env_register.make_env(task_name, seed, maximum_length)
        env.reset()

        file_name = osp.join(root_dir, all_environments[i])

        with open(file_name, 'rb') as pickle_load:
            env_info = pickle.load(pickle_load)

        env.set_info(env_info)
        environment_cache.append(env)

    return environment_cache