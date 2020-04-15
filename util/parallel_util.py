# -----------------------------------------------------------------------------
#   @brief:
#       Define some signals used during parallel
#   @author:
#       Tingwu Wang
# -----------------------------------------------------------------------------

TRAIN_SIGNAL = 1
SAVE_SIGNAL = 2

# it makes the main trpo agent push its weights into the tunnel
START_SIGNAL = 3

# it ends the training
END_SIGNAL = 4

# ends the rollout
END_ROLLOUT_SIGNAL = 5

# ask the rollout agents to collect the ob normalizer's info
AGENT_COLLECT_FILTER_INFO = 6

# ask the rollout agents to synchronize the ob normalizer's info
AGENT_SYNCHRONIZE_FILTER = 7

# ask the agents to set their parameters of network
AGENT_SET_WEIGHTS = 8
AGENT_SET_WEIGHTS_MULTI = 18

# reset
RESET_SIGNAL = 9

# set trainer weights
TRAINER_SET_WEIGHTS = 12

# Initial training for mbmf policy netwrok.
MBMF_INITIAL = 666

# ask for policy network.
GET_POLICY_NETWORK = 6666

# ask and set for policy network weight.
GET_POLICY_WEIGHT = 66
SET_POLICY_WEIGHT = 66666

WORKER_RUNNING = 9
WORKER_PLAYING = 10
WORKER_GET_MODEL = 11
WORKER_SET_ENVIRONMENTS = 13
TRAINER_SET_ENVIRONMENTS = 14

AGENT_RENDER = 15

FETCH_SPARSE_WEIGHTS = 16

FETCH_FINAL_WEIGHTS = 17

WORKER_RUNNING_MT = 19