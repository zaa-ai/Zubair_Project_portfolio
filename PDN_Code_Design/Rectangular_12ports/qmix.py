import numpy as np 
import os 
import argparse
import gymnasium as gym 
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete, Box
import ray
import time
import pandas as pd
from collections import Counter
from ray import tune
from ray.tune import register_env
from ray.rllib.algorithms.qmix.qmix_policy import ENV_STATE
from ray.tune.syncer import SyncConfig
from Pcb_env import PCBEnvironment,PCBWithGroupedAgents
# Import the necessary QMix configuration class
from ray.rllib.algorithms.qmix import QMixConfig


# path and file name of batch file

state_einheit=12
state_num = int(2*state_einheit)

############################################################################################################

pos_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
num_pos = len(pos_list)

class MyCallback(tune.Callback):
    def on_trial_result(self, iteration, trials, trial, result, **info):
        print(f"Iteration: {iteration}, Trial: {trial.trial_id}, Mean Reward: {result['episode_reward_mean']}")


user_name = r"C:\Users\emago"
trial_dir = user_name + r"\PI_PG_WI2324\Square_12ports\result_Qmix"

parser = argparse.ArgumentParser()
parser.add_argument("--run", type=str, default="QMIX")
parser.add_argument("--num-cpus", type=int, default=2)
parser.add_argument("--as-test", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--stop-reward", type=float, default=10)
parser.add_argument("--stop-timesteps", type=int, default=50)

if __name__ == "__main__":
    args = parser.parse_args()

    grouping = {
        "group_1": [0, 1],
    }
    obs_space = Tuple([
        Dict({
            "obs": MultiDiscrete(np.full(len(pos_list)+1, 3), dtype=int),
            ENV_STATE: MultiDiscrete(np.full(len(pos_list), 3), dtype=int)
        }),
        Dict({
            "obs": MultiDiscrete(np.full(len(pos_list)+1, 3), dtype=int),
            ENV_STATE: MultiDiscrete(np.full(len(pos_list), 3), dtype=int)
        }),
    ])
    act_space = Tuple([
        PCBEnvironment.action_space,
        PCBEnvironment.action_space,
    ])
    register_env(
        "grouped_PCB",
        lambda config: PCBWithGroupedAgents(config).with_agent_groups(
            grouping, obs_space=obs_space, act_space=act_space))
    
    single_obs_space = Dict({
        "obs": MultiDiscrete(np.full(len(pos_list)+1, 3), dtype=int),
        ENV_STATE: MultiDiscrete(np.full(len(pos_list), 3), dtype=int)
    })

    # Now define the observation space for each agent, if they are different adjust accordingly
    obs_space_dict = {
        "agent_1": single_obs_space,
        "agent_2": single_obs_space,
    }
    action_space_dict = {
        "agent_1": PCBEnvironment.action_space,
        "agent_2": PCBEnvironment.action_space,
    }




    # Configure the QMix algorithm using the object-oriented approach
    qmix_config = QMixConfig()
    qmix_config.training(
        mixer="qmix",
        mixing_embed_dim=32,
        double_q=True,
        optim_alpha=0.99,
        optim_eps=0.00001,
        grad_clip=10.0,
        lr=0.0005,
        train_batch_size=32,
        target_network_update_freq=5,
        num_steps_sampled_before_learning_starts=10
    ).resources(
        num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")),
        # num_workers=1
    ).rollouts(
        rollout_fragment_length=4,
        batch_mode="complete_episodes"
    )

    
# Since the exploration method call did not work, configure exploration settings directly
    config_dict = qmix_config.to_dict()
    config_dict['exploration_config'] = {
        "type": "EpsilonGreedy",
        "initial_epsilon": 1.0,
        "final_epsilon": 0.1,
        "epsilon_timesteps": 40
    }   

    config_dict.update({
        "env": "grouped_PCB",
        "env_config": {
            "separate_state_space": True,
            "one_hot_state_encoding": True
        },
        "multiagent": {
        "policies": {
                "pol1": (None, Tuple([ single_obs_space, single_obs_space]),
                        Tuple([PCBEnvironment.action_space, PCBEnvironment.action_space]), {
                    "agent_id": 0,
                }),
                "pol2": (None, Tuple([single_obs_space, single_obs_space]),
                        Tuple([PCBEnvironment.action_space, PCBEnvironment.action_space]), {
                    "agent_id": 1,
                }),
            },
            "policy_mapping_fn": lambda agent_id, *args, **kwargs: "pol1" if agent_id == "agent_1" else "pol2",
            "observations": obs_space_dict
        }
    })

    group = True
    start_time = time.time()
    ray.init(num_cpus=args.num_cpus or None)

    stop = {
        "episode_reward_mean": args.stop_reward,
        "timesteps_total": args.stop_timesteps,
    }

    config = dict(config_dict, **{
        "env": "grouped_PCB" if group else PCBEnvironment,
    })
    print("Aqui comienza!!!!!!!!!!!")

    results = tune.run(
        "QMIX",
        stop={"training_iteration": 150},
        config=config_dict,
        storage_path=trial_dir,
        callbacks=[MyCallback()],
        checkpoint_freq=10,  # Save a checkpoint every 10 iterations
        checkpoint_at_end=True, 
        verbose=3,
        sync_config=SyncConfig(sync_timeout=3600)
    )
    end_time = time.time()
    duration = end_time - start_time
    print("training complete in", duration ,"second")
    print("training complete in", duration/60, "minutes")
    print("training complete in", duration/3600, "hours")
    ray.shutdown()
    csv_file_path = user_name + r"\PI_PG_WI2324\Square_12ports\Qmix_termination_state_150_R.csv"
# Load the CSV file into a DataFrame
    df = pd.read_csv(csv_file_path)

    # Extract the states column (assuming it's the second column, index 1)
    states = df['state']
    rewards = df['rewards']

    # Convert states from string to list
    states = states.apply(lambda x: tuple(eval(x.replace(' ', ', '))))
    rewards = rewards.apply(lambda x: max(map(float, eval(x).values())))

    # Count the frequency of each state
    state_counter = Counter(states)

    # Find the most common state
    most_common_state = state_counter.most_common(1)[0]
    best_reward_index = rewards.idxmax()
    best_reward_state = states[best_reward_index]
    best_reward_value = rewards[best_reward_index]


    print(f"The most frequent state is {most_common_state[0]} which appears {most_common_state[1]} times.")
    print(f"The state with the best reward is {best_reward_state} with a reward of {best_reward_value}.")