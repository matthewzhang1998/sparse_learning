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

from trainer import trainer
from runner import ppo_runner
from runner.worker import base_worker
from policy import ppo_policy
from util import init_path

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

    os.makedirs(path)
    return path

def make_trainer(trainer, network_type, params, scope="trainer"):
    # initialized the weights for policy networks and dynamics network

    trainer_tasks = multiprocessing.JoinableQueue()
    trainer_results = multiprocessing.Queue()
    trainer_agent = trainer(params, network_type,
                            trainer_tasks, trainer_results,
                            scope)
    trainer_agent.start()
    # trainer_agent.run()
    trainer_tasks.put((parallel_util.START_SIGNAL, None))

    trainer_tasks.join()

    # init_weights: {'policy': list of weights, 'dynamics': ..., 'reward': ...}
    init_weights = trainer_results.get()
    return trainer_tasks, trainer_results, trainer_agent, init_weights


def make_sampler(sampler, worker_type, network_type, params):
    sampler_agent = sampler.sampler(params, worker_type, network_type)
    return sampler_agent


def log_results(results, timer_dict, start_timesteps=0):
    logger.info("-" * 15 + " Iteration %d " % results['iteration'] + "-" * 15)

    for i_id in range(len(timer_dict) - 1):
        start_key, end_key = list(timer_dict.keys())[i_id: i_id + 2]
        time_elapsed = (timer_dict[end_key] - timer_dict[start_key]) / 60.0

        logger.info("Time elapsed for [{}] is ".format(end_key) +
                    "%.4f mins" % time_elapsed)

    logger.info("{} total steps have happened".format(results['totalsteps']))

    # the stats
    from tensorboard_logger import log_value
    for key in results['stats']:
        logger.info("[{}]: {}".format(key, results['stats'][key]))
        if results['stats'][key] is not None:
            log_value(key, results['stats'][key], start_timesteps +
                      results['totalsteps'])


def train(trainer, sampler, worker, dynamics, policy, reward, params=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))
    network_type = {'policy': policy, 'dynamics': dynamics, 'reward': reward}

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, params)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, params)
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    totalsteps = 0
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

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
        log_results(training_return, timer_dict)

        if totalsteps > params.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))



def train(trainer, sampler, worker, network_type, params=None):
    logger.info('Training starts at {}'.format(init_path.get_abs_base_dir()))

    # make the trainer and sampler
    sampler_agent = make_sampler(sampler, worker, network_type, params)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, params)
    sampler_agent.set_weights(init_weights)

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = \
            sampler_agent._rollout_with_workers()

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}

        if params.pretrain_vae and current_iteration < params.pretrain_iterations:
            training_info['train_net'] = 'vae'

        elif params.decoupled_managers:
            if (current_iteration % \
                (params.manager_updates + params.actor_updates)) \
                < params.manager_updates:
                training_info['train_net'] = 'manager'

            else:
                training_info['train_net'] = 'actor'

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
        log_results(training_return, timer_dict)

        #if totalsteps > params.max_timesteps:
        if training_return['totalsteps'] > params.max_timesteps:
            break
        else:
            current_iteration += 1

    # end of training
    sampler_agent.end()
    trainer_tasks.put((parallel_util.END_SIGNAL, None))

def main():
    parser = base_config.get_base_config()
    params = base_config.make_parser(parser)

    dir = get_dir(params.output_dir)

    if params.write_log:
        logger.set_file_handler(path=params.output_dir,
                                prefix='sparse_baseline' + params.task,
                                time_str=params.exp_id)

    argparse_dict = vars(params)
    import json
    import os.path as osp
    with open(osp.join(params.output_dir, 'args.json'), 'w') as f:
        json.dump(argparse_dict, f)

    print('Training starts at {}'.format(init_path.get_abs_base_dir()))

    train(trainer.Trainer, ppo_runner.Sampler, base_worker.Worker,
          ppo_policy.PPOPolicy, params)

if __name__ == '__main__':
    main()