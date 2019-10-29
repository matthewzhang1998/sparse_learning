#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 19:53:57 2018
@author: matthewszhang
"""

import importlib
import re

_ENV_INFO = {
    'point_composition': {
        'path': 'point_composite',
        'ob_size': 4, 'action_size': 8,
        'action_distribution': 'discrete'
    }
}


def io_information(task_name):
    render_flag = re.compile(r'__render$')
    task_name = render_flag.sub('', task_name)

    return _ENV_INFO[task_name]['ob_size'], \
           _ENV_INFO[task_name]['action_size'], \
           _ENV_INFO[task_name]['action_distribution']


def make_env(task_name, rand_seed, maximum_length, misc_info=None):
    env_file = importlib.import_module(_ENV_INFO[task_name]['path'])
    return env_file.Env(task_name, rand_seed, maximum_length, misc_info), \
           _ENV_INFO[task_name]


