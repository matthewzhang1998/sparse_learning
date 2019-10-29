#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 11:10:37 2018
@author: matthewszhang
"""
import tensorflow as tf
import numpy as np

from util import tf_util

class BasePolicy(object):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError("")

    def get_weights(self):
        return self._get_network_weights()

    def set_weights(self, weights):
        self._set_network_weights(weights)

    def _set_var_list(self):
        # collect the tf variable and the trainable tf variable
        self._trainable_var_list = [var for var in tf.trainable_variables()
                                    if self._name_scope in var.name]

        self._all_var_list = [var for var in tf.global_variables()
                              if self._name_scope in var.name]

        # the weights that actually matter
        self._network_var_list = \
            self._trainable_var_list  # + self._whitening_variable

        self._set_network_weights = tf_util.set_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

        self._get_network_weights = tf_util.get_network_weights(
            self._session, self._network_var_list, self._name_scope
        )

    # null methods covered by __init__
    def build_model(self):
        return

    def get_input_placeholder(self):
        return {}

    def _discard_final_states(self, data_dict):
        # remove terminated states from dictionary
        for key in data_dict:
            _temp_item = np.array(data_dict[key])

            if _temp_item.ndim != 0:
                # if the shape divides by the episode length + 1
                if (_temp_item.shape[0] \
                    % (self.args.episode_length + 1)) == 0:

                    _temp_data = np.reshape(_temp_item,
                                            [-1, self.args.episode_length + 1,
                                             *_temp_item.shape[1:]])

                    if key in ['motivations']:
                        _temp_data = _temp_data[:, 1:]
                    else:
                        _temp_data = _temp_data[:, :-1]
                    data_dict[key] = np.reshape(_temp_data,
                                                [-1, *_temp_item.shape[1:]]
                                                )

        return data_dict