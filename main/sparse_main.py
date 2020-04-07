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
from policy import sparse_ppo_policy, ppo_policy, consolidated_ppo_policy
from util import init_path
import tensorboard_logger

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

def make_trainer(trainer, network_type, params, scope="trainer", init_weights=None, path=None):
    # initialized the weights for policy networks and dynamics network

    trainer_tasks = multiprocessing.JoinableQueue()
    trainer_results = multiprocessing.Queue()
    trainer_agent = trainer(params, network_type,
                            trainer_tasks, trainer_results,
                            scope, init_weights=init_weights, path=path)
    trainer_agent.start()
    # trainer_agent.run()
    trainer_tasks.put((parallel_util.START_SIGNAL, None))

    trainer_tasks.join()

    # init_weights: {'policy': list of weights, 'dynamics': ..., 'reward': ...}
    init_weights = trainer_results.get()
    return trainer_tasks, trainer_results, trainer_agent, init_weights


def make_sampler(sampler, worker_type, network_type, params, subtask=None):
    sampler_agent = sampler.Sampler(params, worker_type, network_type, subtask=subtask)
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

    sampler_agent = []

    for i in range(params.num_subtasks):
        sampler_agent.append(make_sampler(sampler, worker, dense_network_type, params, subtask=i))

    if params.separate_train:

        trainer_tasks = []
        trainer_results = []
        trainer_agent = []
        init_weights = []

        for i in range(params.num_subtasks):
            tasks, results, agent, weights = \
                make_trainer(trainer, network_type, params, path=path)

            trainer_tasks.append(tasks)
            trainer_results.append(results)
            trainer_agent.append(agent)
            init_weights.append(weights)

    else:
        trainer_tasks, trainer_results, trainer_agent, init_weights = \
            make_trainer(trainer, network_type, params, path=path)

    for i in range(params.num_subtasks):
        sampler_agent[i].set_weights(init_weights[i])

    timer_dict = OrderedDict()
    timer_dict['Program Start'] = time.time()
    current_iteration = 0

    max_pre_iter = (((1 - params.sparsification_floor) //
        params.sparsification_percent )+1) * params.sparsification_iter

    print("MAX", max_pre_iter, flush=True)

    while True:
        timer_dict['** Program Total Time **'] = time.time()

        # step 1: collect rollout data
        rollout_data = []
        for i in range(params.num_subtasks):
            rollout_data.append(
                sampler_agent[i]._rollout_with_workers())

        timer_dict['Generate Rollout'] = time.time()

        # step 2: train the weights for dynamics and policy network
        training_info = {}

        if params.separate_train:
            for i in range(params.num_subtasks):
                trainer_tasks[i].put(
                    (parallel_util.TRAIN_SIGNAL,
                     {'data': rollout_data[i]['data'], 'training_info': training_info})
                )

            for i in range(params.num_subtasks):
                trainer_tasks[i].join()

            training_return = []
            for i in range(params.num_subtasks):
                training_return.append(trainer_results[i].get())

        else:
            trainer_tasks.put(
                (parallel_util.TRAIN_SIGNAL,
                 {'data': [rollout_data[i]['data'] for i in range(params.num_subtasks)],
                  'training_info': training_info})
            )

            trainer_tasks.join()

            training_return = trainer_results.get()

        print(type(training_return))

        timer_dict['Train Weights'] = time.time()

        if current_iteration % params.render_iter == 0:
            for i in range(params.num_subtasks):
                print("_____RENDERING_____")
                sampler_agent[i].render(current_iteration, save_loc + '/' + str(i))

        # step 4: update the weights
        for i in range(params.num_subtasks):
            sampler_agent[i].set_weights(training_return[i]['network_weights'])

        timer_dict['Assign Weights'] = time.time()

        # log and print the results

        for i in range(params.num_subtasks):
            log_results(training_return[i], timer_dict, tb_logger=tb_logger[i])

        #if totalsteps > params.max_timesteps:
        if training_return[0]['totalsteps'] > params.max_timesteps:
            break
        elif current_iteration - 1 > max_pre_iter:
            break
        else:
            current_iteration += 1

    # end of training
    if params.separate_train:
        init_policy_weights = []
        init_value_weights = []

        for i in range(params.num_subtasks):
            trainer_tasks[i].put(
                (parallel_util.FETCH_SPARSE_WEIGHTS, {})
            )

        for i in range(params.num_subtasks):
            trainer_tasks[i].join()

        for i in range(params.num_subtasks):
            p, v = trainer_results[i].get()
            init_policy_weights.append(p)
            init_value_weights.append(v)

    else:
        trainer_tasks.put(
            (parallel_util.FETCH_SPARSE_WEIGHTS, {})
        )

        trainer_tasks.join()

        init_policy_weights, init_value_weights= trainer_results.get()

    fin_policy_weights = []
    fin_value_weights = []
    for i in range(len(init_policy_weights[0])):
        w = sum([x[i][0] for x in init_policy_weights])
        b = sum([x[i][1] for x in init_policy_weights])
        fin_policy_weights.append((w,b))
    for j in range(len(init_value_weights[0])):
        w = sum([x[j][0] for x in init_value_weights])
        b = sum([x[j][1] for x in init_value_weights])
        fin_value_weights.append((w,b))

    for i in range(params.num_subtasks):
        sampler_agent[i].end()

    if params.separate_train:
        for i in range(params.num_subtasks):
            trainer_tasks[i].put((parallel_util.END_SIGNAL, None))
    else:
        trainer_tasks.put((parallel_util.END_SIGNAL, None))

    # make the trainer and sampler
    fin_weights = {'policy': fin_policy_weights, 'value': fin_value_weights}

    sampler_agent = make_sampler(sampler, worker, dense_network_type, params)
    trainer_tasks, trainer_results, trainer_agent, init_weights = \
        make_trainer(trainer, network_type, params,
                     init_weights=fin_weights, path=path)

    sampler_agent.set_weights(init_weights)

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

        if current_iteration % params.render_iter == 0:
            print("_____RENDERING_____")
            sampler_agent.render(current_iteration, save_loc + '/' + 'dense')

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
        train(trainer.Trainer, ppo_runner, base_worker,
              sparse_ppo_policy.SparsePPOPolicy, ppo_policy.PPOPolicy, params)

    else:
        train(trainer.Trainer, ppo_runner, base_worker,
              consolidated_ppo_policy.ConsolidatedPPOPolicy, ppo_policy.PPOPolicy, params)

if __name__ == '__main__':
    main()