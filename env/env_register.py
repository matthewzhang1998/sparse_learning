#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:53:57 2018
@author: matthewszhang
"""

import importlib
import importlib.util
import re
import os
import os.path as osp

import metaworld
import numpy as np

_ENV_INFO = {
    'point_composite': {
        'path': 'point_composite.py',
        'ob_size': 9, 'action_size': 2,
        'action_distribution': 'continuous'
    },
    'robot_composite': {
        'path': 'robot_composite.py',
        'ob_size': 29, 'action_size': 4,
        'action_distribution': 'continuous'
    }
}


def io_information(task_name):
    render_flag = re.compile(r'__render$')
    task_name = render_flag.sub('', task_name)

    return _ENV_INFO[task_name]['ob_size'], \
           _ENV_INFO[task_name]['action_size'], \
           _ENV_INFO[task_name]['action_distribution']


def mt_io_information(n_subtasks):
    if n_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_train_tasks()
        task = env.sample_tasks(1)
        env.set_task(task[0])
        return np.prod(env.observation_space.shape),\
               np.prod(env.action_space.shape), \
            'continuous'

    else:
        env = metaworld.benchmarks.ML50.get_train_tasks()
        task = env.sample_tasks(1)
        env.set_task(task[0])
        return np.prod(env.observation_space.shape),\
               np.prod(env.action_space.shape), \
            'continuous'


def make_env(task_name, rand_seed, maximum_length, misc_info=None):
    dir_path = osp.dirname(osp.abspath(__file__))

    env_file = importlib.util.spec_from_file_location("Env",
        osp.join(dir_path,_ENV_INFO[task_name]['path']))
    env = importlib.util.module_from_spec(env_file)
    env_file.loader.exec_module(env)

    return env.Env(task_name, rand_seed, maximum_length, misc_info), \
           _ENV_INFO[task_name]

def make_mt_env(task_name, num_subtasks):
    if num_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_train_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    else:
        env = metaworld.benchmarks.ML50.get_train_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    return env


def make_mt_test(task_name, num_subtasks):
    if num_subtasks <= 10:
        env = metaworld.benchmarks.ML10.get_test_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    else:
        env = metaworld.benchmarks.ML50.get_test_tasks()
        tasks = env.sample_tasks(num_subtasks)
        env.set_task(tasks[task_name])

    return env



