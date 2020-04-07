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


def make_env(task_name, rand_seed, maximum_length, misc_info=None):
    dir_path = osp.dirname(osp.abspath(__file__))

    env_file = importlib.util.spec_from_file_location("Env",
        osp.join(dir_path,_ENV_INFO[task_name]['path']))
    env = importlib.util.module_from_spec(env_file)
    env_file.loader.exec_module(env)

    return env.Env(task_name, rand_seed, maximum_length, misc_info), \
           _ENV_INFO[task_name]


