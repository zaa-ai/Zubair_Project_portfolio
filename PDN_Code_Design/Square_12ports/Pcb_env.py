import numpy as np 
import help_qmix as fh
import xml.etree.ElementTree as ET
import csv
import sys
import json
import pickle
from gym.spaces import Tuple, MultiDiscrete, Dict, Discrete
from ray.rllib.env.multi_agent_env import MultiAgentEnv, make_multi_agent
from ray.rllib.algorithms.qmix.qmix_policy import ENV_STATE


# path and file name of batch file
user_name = r"C:\Users\emago"
source_file = user_name + r"\PI_PG_WI2324\Square_12port\Batch_12Decaps_Values.peb"
destination_file = user_name + r"\PI_PG_WI2324\Square_12port\tmp_Batch_12Decaps_Values.peb"


############################################################################################################
state_einheit=12
state_num = int(2*state_einheit)

############################################################################################################

pos_list = np.array([1,2,3,4,5,6,7,8,9,10,11,12])
num_pos = len(pos_list)

# decap Type
type1 = [100e-9 , 222e-12 , 8.9e-3]
type_list = np.array([1])
num_type = len(type_list)

# define parameters
fmin=2e5
fmax=2e8
fmittel=1e7
Zmin=15
Zmax= 90
###########################################################################################################
result_list = []
reward_list = []
episode_num = []
decaps_num =  []
critic_loss_list = []
actor_po_loss_list = []
actor_type_loss_list = []
loss_count_list = []


##########################################################################################
#define function for class PCB
##########################################################################################
def reset_pebfiles():
    source = open(source_file)
    tree = ET.parse(source)
    tree.write(destination_file)
    source.close()

# peb.file generation
# peb.file generation
def gen_pebfiles(action):
    position = action[0]
    value = action[1]

    source = open(destination_file)
    tree = ET.parse(source)
    root = tree.getroot()

    k = position - 1

    if value == 0:
        print('No Action has been selected!')
        sys.exit()
    elif value == 1:
        root[0][0][4 * k].set('Value', 'true')
        root[0][0][4 * k + 1].set('Value', str(type1[0]))
        root[0][0][4 * k + 2].set('Value', str(type1[2]))
        root[0][0][4 * k + 3].set('Value', str(type1[1]))

    tree.write(destination_file)
    source.close()
# read current impedance and reward funktion
# global variables
num_t = 0  # the number of frequency points satisfying the target impedance at the steps t
num_t1 = 0  # the number of frequency points satisfying the target impedance at the steps t+1

def cal_reward(state,agent):
    ################ read each Csv file #################################
    global num_t, num_t1
    # with open('/content/sample1'+ str(file_number+1) +'.csv', newline='') as csvfile:
    used_decap = sum(state)
    if agent == 0:
        with open(user_name + r"\PI_PG_WI2324\Square_12port\Design1.emc\PI-1/Power_GND/1-PIPinZ_IC1.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)
            current_impedance_list = list(reader)
        frequency, impedance_values, num_rows1=fh.get_freqency_and_impedance(current_impedance_list)
        ######################### count #############################
        count = 0  # how many points satisfy the target impedance
        for i in range(num_rows1):
            if impedance_values[i] <= fh.target_impedance(frequency[i], fmin, fmax, fmittel, Zmin, Zmax):
                # datatype
                count += 1

        num_t = num_t1
        num_t1 = count
        empty_decap = state_einheit - used_decap
        if num_t1 == num_rows1:
            return ((num_t1 - num_t) / (num_rows1)) + (10 * (empty_decap / state_einheit)), True
        else:
            return (num_t1 - num_t) / (num_rows1), False
    if agent == 1:
        with open(user_name + r"\PI_PG_WI2324\Square_12port\Design1.emc\PI-1/Power_GND/1-PIPinZ_IC2.csv", newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            next(reader, None)
            current_impedance_list = list(reader)
        frequency, impedance_values, num_rows2=fh.get_freqency_and_impedance(current_impedance_list)
        ######################### count #############################
        count_2 = 0  # how many points satisfy the target impedance
        for i in range(num_rows2):
            if impedance_values[i] <= fh.target_impedance(frequency[i], fmin, fmax, fmittel, Zmin, Zmax):
                # datatype
                count_2 += 1        
        num_t = num_t1
        num_t1 = count_2
        empty_decap = state_einheit - used_decap
        if num_t1 == num_rows2:
            return ((num_t1 - num_t) / (num_rows2)) + (10 * (empty_decap / state_einheit)), True
        else:
            return (num_t1 - num_t) / (num_rows2), False 
    

# Define the PCBEnvironment class as before


class PCBEnvironment(MultiAgentEnv):

    action_list=fh.possible_action(pos_list, type_list)
    action_space =  Discrete(len(action_list))
    observation_space =  MultiDiscrete(np.full(len(pos_list),3),dtype=int)
    def __init__(self, env_config):
        super().__init__()
        self.action_pos = pos_list
        self.action_type = type_list
        self.action_list=fh.possible_action(pos_list, type_list)
        self.action_space =  Discrete(len(self.action_list))
        self.state = None
        self.C_ou = 0
        self.agent_1 = 0
        self.agent_2 = 1
        self._skip_env_checking = True
        self.preobs = np.zeros((len(pos_list)),dtype=int)
        self.actions_are_logits = env_config.get("actions_are_logits", False)
        self.one_hot_state_encoding = env_config.get("one_hot_state_encoding", False)
        self.with_state = env_config.get("separate_state_space", False)
        self._agent_ids = {0, 1}
        if not self.one_hot_state_encoding:
            self.observation_space =  MultiDiscrete(np.full(len(pos_list),3),dtype=int)
            self.with_state = False
        else:
            if self.with_state:
                self.observation_space = Dict(
                    {
                        "obs": MultiDiscrete(np.full(len(pos_list),3),dtype=int),
                        ENV_STATE: MultiDiscrete(np.full(len(pos_list),3),dtype=int)
                    }
                )
            else:
                self.observation_space =  MultiDiscrete(np.full(len(pos_list),3),dtype=int)

        # every new episode
    def reset(self,seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        global num_t, num_t1
        num_t = 0
        num_t1 = 0
        self.count = 0
        self.state= np.zeros((len(pos_list)),dtype=int)
        reset_pebfiles()
        return self._obs(), {}

    def step(self, action_dict):
        # initialize the current state after the selection of action
        actiona = [0 , 1]
        action = []
        action_list=fh.possible_action(pos_list, type_list)
        for i in range(2):
            actiona[i] = action_dict[i]
            action.append(action_list[actiona[i]])
            gen_pebfiles(action[i])
        next_obs = fh.next_state_step(self.state, action[0])
        self.state = next_obs  # Be cautious about shared state across agents
        next_obs = fh.next_state_step(self.state, action[1])
        self.state = next_obs  # Be cautious about shared state across agents
        # self.current_obs = self.state  # This might also need to be specific per agent
        self.preobs = next_obs
        # Handle file operations related to the action
        

        # Compute reward, check if the episode is done, etc.
     
        fh.load_batch_file(destination_file)
        Reward_value1, done1 = cal_reward(next_obs,0)
        Reward_value2, done2 = cal_reward(next_obs,1)
        
        rewards_dict = {self.agent_1: Reward_value1, self.agent_2: Reward_value2}
        self.count += 1
        obs = self._obs()
        terminateds = {self.agent_1: done1, self.agent_2: done2, "__all__": done1 and done2}
        truncateds = {"__all__": False}
        infos = {
            self.agent_1: {"done": terminateds[self.agent_1]},
            self.agent_2: {"done": terminateds[self.agent_2]},
        }
        self.C_ou += 1
        print(infos)
 ####################################################################       
        if terminateds["__all__"]:
            # Convert complex objects to strings or flatten them as needed
            state_str = str(self.state)  # Example conversion, adjust as needed
            rewards_str = {k: str(v) for k, v in rewards_dict.items()}  # Convert each value to string

            data_to_save = {
                'iteration': self.C_ou,
                'state': state_str,
                'rewards': rewards_str,
                'count': self.count
            }

            # Define the CSV file path
            csv_file_path = user_name + r"\PI_PG_WI2324\Square_12port\Qmix_termination_state_150.csv"

            # Write to CSV, appending each time
            with open(csv_file_path, 'a', newline='') as file:
                fieldnames = ['iteration', 'state', 'rewards', 'count']
                writer = csv.DictWriter(file, fieldnames=fieldnames)

                # Write headers if the file is newly created
                if file.tell() == 0:
                    writer.writeheader()

                writer.writerow(data_to_save)

            print("State appended to CSV file due to termination of both agents.")
######################################################################################
        if sum(self.preobs)>state_einheit:
            reset_pebfiles()
        return obs, rewards_dict, terminateds, truncateds, infos
    
    def _obs(self):
        if self.with_state:
            
            obs_space = {
                self.agent_1: {"obs": self.agent_1_obs(), ENV_STATE: self.state},
                self.agent_2: {"obs": self.agent_2_obs(), ENV_STATE: self.state},
            }
 
            return obs_space 
        else:
            return {self.agent_1: self.agent_1_obs(), self.agent_2: self.agent_2_obs()}

        
    def agent_1_obs(self):
        
        if self.one_hot_state_encoding:
            return np.concatenate([self.state, [1]])
        else:
            return np.flatnonzero(self.state)[0]

    def agent_2_obs(self):
        if self.one_hot_state_encoding:
            return np.concatenate([self.state, [2]])
        else:
            return np.flatnonzero(self.state)[0] + 3
        


class PCBWithGroupedAgents(MultiAgentEnv):
    def __init__(self, env_config):
        super().__init__()
        env = PCBEnvironment(env_config)
        tuple_obs_space = Tuple([env.observation_space, env.observation_space])
        tuple_act_space = Tuple([env.action_space, env.action_space])

        self.env = env.with_agent_groups(
            groups={"agents": [0, 1]},
            obs_space=tuple_obs_space,
            act_space=tuple_act_space,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = {"agents"}
        self._skip_env_checking = True
    
    def reset(self, *, seed=None, options=None):
        return self.env.reset(seed=seed, options=options)
    
    def step(self, actions):
        return self.env.step(actions)
    

