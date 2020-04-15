
import argparse
from util import init_path

def get_base_config():

    # get the parameters
    parser = argparse.ArgumentParser(description='Sparse Composite Tasks')

    # the experiment settings
    parser.add_argument("--task", type=str,
                        default='robot_composite',
                        help='the environment to test')
    parser.add_argument("--exp_id", type=str, default="log/composite",
                        help='the special id of the experiment')
    parser.add_argument("--episode_length", type=int, default=100,
                        help='length of the environment')
    parser.add_argument("--gamma", type=float, default=.05,
                        help='the discount factor for value function')
    parser.add_argument("--seed", type=int, default=1234)

    # training configuration
    parser.add_argument("--batch_size", type=int, default=5000,
                        help='number of steps in the rollout')
    parser.add_argument("--max_timesteps", type=int, default=1e9)
    parser.add_argument("--num_minibatches", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=5)

    parser.add_argument("--use_replay_buffer", type=int, default=0)
    parser.add_argument("--use_state_normalization", type=int, default=0)
    parser.add_argument("--replay_buffer_size", type=int, default=25000)
    parser.add_argument("--replay_buffer_type", type=str,
                        default='prioritized_by_episode')
    parser.add_argument("--buffer_priority_alpha", type=float,
                        default=1.0)
    parser.add_argument("--replay_batch_size", type=float,
                        default=5000)

    parser.add_argument("--cache_environments", type=int, default=0)
    parser.add_argument("--load_environments", type=str, default=None)
    parser.add_argument("--num_cache", type=int, default=1)

    parser.add_argument("--policy_lr", type=float, default=3e-4)
    parser.add_argument("--policy_epochs", type=int, default=5)
    parser.add_argument("--policy_network_shape", type=str, default='128,128,128')
    parser.add_argument("--policy_activation_type", type=str, default='tanh')
    parser.add_argument("--policy_normalizer_type", type=str,
                        default='none')
    parser.add_argument("--gae_lam", type=float, default=0.95)
    parser.add_argument("--fisher_cg_damping", type=float, default=0.1)
    parser.add_argument("--target_kl", type=float, default=0.01)
    parser.add_argument("--cg_iterations", type=int, default=10)

    parser.add_argument("--ppo_clip", type=float, default=0.1)
    parser.add_argument("--target_kl_high", type=float, default=2)
    parser.add_argument("--target_kl_low", type=float, default=0.5)
    parser.add_argument("--use_weight_decay", type=int, default=0)
    parser.add_argument("--weight_decay_coeff", type=float, default=1e-5)

    parser.add_argument("--use_kl_penalty", type=int, default=0)
    parser.add_argument("--kl_alpha", type=float, default=1.5)
    parser.add_argument("--kl_eta", type=float, default=50)

    parser.add_argument("--policy_lr_schedule", type=str, default='linear',
                        help='["linear", "constant", "adaptive"]')
    parser.add_argument("--policy_lr_alpha", type=int, default=2)

    parser.add_argument("--joint_value_update", type=int, default=0)

    parser.add_argument("--clip_gradients", type=int, default=1)
    parser.add_argument("--clip_gradient_threshold", type=float, default=.1)

    parser.add_argument("--value_lr", type=float, default=3e-4)
    parser.add_argument("--value_epochs", type=int, default=10)
    parser.add_argument("--value_network_shape", type=str, default='128,128')
    parser.add_argument("--value_activation_type", type=str, default='relu')
    parser.add_argument("--value_normalizer_type", type=str,
                        default='none')

    # the checkpoint and summary setting
    parser.add_argument("--ckpt_name", type=str, default=None)
    parser.add_argument("--output_dir", '-o', type=str, default='composite')
    parser.add_argument('--write_log', type=int, default=1)

    # debug setting
    parser.add_argument("--monitor", type=int, default=0)
    parser.add_argument("--debug", type=int, default=0)

    parser.add_argument("--render_iter", type=int, default=10)
    parser.add_argument("--render_save_loc", type=str, default="./render/")

    parser.add_argument("--num_subtasks", type=int, default=10)

    parser.add_argument("--sparsification_iter", type=int, default=50)
    parser.add_argument("--sparsification_percent", type=float, default=0.1)
    parser.add_argument("--sparsification_floor", type=float, default=0.5)

    parser.add_argument("--max_iter", type=int, default=5e4)
    parser.add_argument("--separate_train", type=int, default=0)

    parser.add_argument("--use_subtask_value", type=int, default=1)
    parser.add_argument("--mask_penalty", type=float, default=1e-4)
    parser.add_argument("--correlation_coefficient", type=float, default=1e-2)

    parser.add_argument("--expansion_coeff", type=int, default = 5)

    return parser

def make_parser(parser):
    return post_process(parser.parse_args())

def post_process(args):
    # parse the network shape
    for key in dir(args):
        if 'shape' in key:
            if getattr(args, key) is None:
                setattr(args, key, [])
            elif 'network' in key:
                setattr(args, key, [int(dim) for dim in getattr(args, key).split(',')])
            else:
                setattr(args, key, [str(dim) for dim in getattr(args, key).split(',')])
    return args