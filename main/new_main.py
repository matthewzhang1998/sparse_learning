"""
Created on Sun Aug 12 23:36:56 2018
@author: matthewszhang
"""
'''
Credit to tingwu wang for implementation
'''
import os
import os.path as osp
import sys

from config import base_config
from util import logger
from util import parallel_util

from trainer import mt_trainer
from runner import ppo_runner
from runner.worker import base_worker
from policy import pretrain_policy, sparse_sac_policy, sac_policy
from util import init_path
import tensorboard_logger

from metaworld.benchmarks import ML10, ML50

import multiprocessing
import time
from collections import OrderedDict

def get_dir(base_path):
    path = base_path
    ix = -1
    while osp.exists(path):
        ix += 1
        path = base_path + str(ix)

    print(osp.join(os.getcwd() + path))
    sys.stdout.flush()

    return path

def make_trainer(trainer, network_type, params, scope="trainer", init_weights=None, path=None,
                 task_names=None):
    # initialized the weights for policy networks and dynamics network

    trainer_tasks = multiprocessing.JoinableQueue()
    trainer_results = multiprocessing.Queue()
    trainer_agent = trainer(params, network_type,
                            trainer_tasks, trainer_results,
                            scope, init_weights=init_weights, path=path,
                            task_names=task_names)
    trainer_agent.start()
    # trainer_agent.run()
    trainer_tasks.put((parallel_util.START_SIGNAL, None))

    trainer_tasks.join()

    # init_weights: {'policy': list of weights, 'dynamics': ..., 'reward': ...}
    init_weights = trainer_results.get()
    return trainer_tasks, trainer_results, trainer_agent, init_weights


def make_sampler(sampler, worker_type, network_type, params, task_names=None):
    sampler_agent = sampler.Sampler(params, worker_type, network_type, task_names=task_names)
    return sampler_agent


def log_results(results, timer_dict, start_timesteps=0, tb_logger=None):
    logger.info("-" * 15 + " Iteration %d " % results['iteration'] + "-" * 15)

    for i_id in range(len(timer_dict) - 1):
        start_key, end_key = list(timer_dict.keys())[i_id: i_id + 2]
        time_elapsed = (timer_dict[end_key] - timer_dict[start_key]) / 60.0

        logger.info("Time elapsed for [{}] is ".format(end_key) +
                    "%.4f mins" % time_elapsed)

    logger.info("{} total steps have happened".format(results['totalsteps']))

    # the stats
    for key in results['stats']:
        logger.info("[{}]: {}".format(key, results['stats'][key]))
        if results['stats'][key] is not None:
            if tb_logger is None:
                tensorboard_logger.log_value(key, results['stats'][key], start_timesteps +
                          results['totalsteps'])
            else:
                tb_logger.log_value(key, results['stats'][key], start_timesteps +
                    results['totalsteps'])

def train(trainer, sampler, worker, network_type, dense_network_type, params=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))

    path = logger.get_tbl_path()

    save_loc = os.path.split(path)[0]
    tb_logger = []
    for i in range(params.num_subtasks):
        os.makedirs(save_loc + '/' + str(i))
        tb_logger.append(tensorboard_logger.Logger(
            os.path.splitext(path)[0] + str(i) + '.log'
        ))
    tb_dense_logger = tensorboard_logger.Logger(path)

    import json
    argparse_dict = vars(params)
    with open(os.path.join(path, 'args.json'), 'w') as f:
        json.dump(argparse_dict, f)

    # make the trainer and sampler

    task_names = [i for i in range(params.num_subtasks)]

    sampler_agent = \
            make_sampler(sampler, worker, dense_network_type, params, task_names)

    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, params, path=path, task_names=task_names)

    sampler_agent.set_weights_multi(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = sampler_agent._rollout_with_workers()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}

        trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
             {'data': [rollout_data[i]['data'] for i in range(params.num_subtasks)],
              'training_info': training_info})
        )

        trainer_tasks.join()

        training_return = trainer_results.get()

        timer_dict['Train Weights'] = time.time()

        weights_dict = {subtask: training_return[subtask]['network_weights']
                for subtask in task_names}

        sampler_agent.set_weights_multi(weights_dict)

        timer_dict['Assign Weights'] = time.time()

        # log and print the results

        for i in range(params.num_subtasks):
            log_results(training_return[i], timer_dict, tb_logger=tb_logger[i])

        #if totalsteps > params.max_timesteps:
        if training_return[0]['totalsteps'] > params.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    trainer_tasks.put(parallel_util.FETCH_FINAL_WEIGHTS)

    trainer_tasks.join()
    final_weights = trainer_results.get()


    for i in range(params.num_subtasks):
        sampler_agent[i].end()

    sampler_agent = make_sampler(sampler, worker, dense_network_type, params, task_names=task_names)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, params, path=path, task_names=task_names)

    sampler_agent.set_weights(final_weights)
    trainer_tasks.put([parallel_util.TRAINER_SET_WEIGHTS, final_weights])

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0
    os.makedirs(save_loc + '/' + 'dense')

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = sampler_agent._rollout_with_workers()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}

        trainer_tasks.put(
            (parallel_util.TRAIN_SIGNAL,
            {'data': rollout_data['data'], 'training_info': training_info})
        )

        trainer_tasks.join()

        training_return = trainer_results.get()

        timer_dict['Train Weights'] = time.time()

        # step 4: update the weights
        sampler_agent.set_weights(training_return['network_weights'])
        timer_dict['Assign Weights'] = time.time()

        # log and print the results

        log_results(training_return, timer_dict, tb_logger=tb_dense_logger)

        # if totalsteps > params.max_timesteps:
        if current_iteration > params.max_iter:
            break
        else:
            current_iteration += 1



def main():
    parser = base_config.get_base_config()
    params = base_config.make_parser(parser)

    dir = osp.join('../log/baseline_' + params.task, params.output_dir)
    dir = get_dir(dir)
    if not osp.exists(dir):
        os.makedirs(dir)

    if params.write_log:
        logger.set_file_handler(dir,
                                time_str=params.exp_id)

    argparse_dict = vars(params)
    import json
    with open(osp.join(dir, 'args.json'), 'w') as f:
        json.dump(argparse_dict, f)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))

    if params.separate_train:
        pass

    else:
        train(mt_trainer.Trainer, ppo_runner, base_worker,
              sparse_sac_policy.SparsePolicy, sac_policy.Policy, params)

if __name__ == '__main__':
    main()