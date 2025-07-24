import pandas as pd
import matplotlib.pyplot as plt
import json
import os
####Code to plot the result obtained by the train ####
# Path to the result,json file this file is storage in .Results
user_name = r"C:\Users\emago\Desktop\PI_MARL"
results_file = os.path.join(user_name, r"PI_PG_WI2324\.Results\square_2decap_dell\result.json")
results_file_1 = os.path.join(user_name, r"PI_PG_WI2324\.Results\rect_1decap_dell\result.json")
results_file_2 = os.path.join(user_name, r"PI_PG_WI2324\.Results\rect_2decap_del\result.json")
# Initialize a list to store the data
data = []
data_1 = []
data_2 = []

# Read the file and parse each JSON object
with open(results_file, 'r') as file:
    for line in file:
        try:
            # Load each JSON object and append to the list
            json_object = json.loads(line)
            data.append(json_object)
        except json.JSONDecodeError:
            print("Error decoding JSON")
##########################################################################
with open(results_file_1, 'r') as file:
    for line in file:
        try:
            # Load each JSON object and append to the list
            json_object = json.loads(line)
            data_1.append(json_object)
        except json.JSONDecodeError:
            print("Error decoding JSON")
###########################################################################
with open(results_file_2, 'r') as file:
    for line in file:
        try:
            # Load each JSON object and append to the list
            json_object = json.loads(line)
            data_2.append(json_object)
        except json.JSONDecodeError:
            print("Error decoding JSON")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)
df_1 = pd.DataFrame(data_1)
df_2 = pd.DataFrame(data_2)
# Helper function to safely extract nested values
def get_nested_value(d, keys, default=None):
    for key in keys:
        d = d.get(key, default)
        if d is default:
            break
    return d

# Extract relevant columns
df_extracted = pd.DataFrame({
    'iteration': range(len(df)),
    'episode_reward_mean': df['sampler_results'].apply(lambda x: x.get('episode_reward_mean')),
    'episode_reward_max': df['sampler_results'].apply(lambda x: x.get('episode_reward_max')),
    'episode_reward_min': df['sampler_results'].apply(lambda x: x.get('episode_reward_min')),
    'num_env_steps_sampled': df['info'].apply(lambda x: x.get('num_env_steps_sampled')),
    'num_env_steps_trained': df['info'].apply(lambda x: x.get('num_env_steps_trained')),
    'loss': df['info'].apply(lambda x: get_nested_value(x, ['learner', 'pol2', 'learner_stats', 'loss'], default=None))
})
#########################################################################################################################
df_extracted_1 = pd.DataFrame({
    'iteration': range(len(df_1)),
    'episode_reward_mean': df_1['sampler_results'].apply(lambda x: x.get('episode_reward_mean')),
    'episode_reward_max': df_1['sampler_results'].apply(lambda x: x.get('episode_reward_max')),
    'episode_reward_min': df_1['sampler_results'].apply(lambda x: x.get('episode_reward_min')),
    'num_env_steps_sampled': df_1['info'].apply(lambda x: x.get('num_env_steps_sampled')),
    'num_env_steps_trained': df_1['info'].apply(lambda x: x.get('num_env_steps_trained')),
    'loss': df_1['info'].apply(lambda x: get_nested_value(x, ['learner', 'pol2', 'learner_stats', 'loss'], default=None))
})

#############################################################################################################################################
df_extracted_2 = pd.DataFrame({
    'iteration': range(len(df_2)),
    'episode_reward_mean': df_2['sampler_results'].apply(lambda x: x.get('episode_reward_mean')),
    'episode_reward_max': df_2['sampler_results'].apply(lambda x: x.get('episode_reward_max')),
    'episode_reward_min': df_2['sampler_results'].apply(lambda x: x.get('episode_reward_min')),
    'num_env_steps_sampled': df_2['info'].apply(lambda x: x.get('num_env_steps_sampled')),
    'num_env_steps_trained': df_2['info'].apply(lambda x: x.get('num_env_steps_trained')),
    'loss': df_2['info'].apply(lambda x: get_nested_value(x, ['learner', 'pol2', 'learner_stats', 'loss'], default=None))
})

# Plotting episode reward mean over iterations
plt.figure(figsize=(10, 5))
plt.plot(df_extracted['iteration'], df_extracted['episode_reward_mean'], marker='o', linestyle='-', color='b', label='Square')
plt.plot(df_extracted_1['iteration'], df_extracted_1['episode_reward_mean'], marker='o', linestyle='-', color='g', label='Rectangular')
plt.plot(df_extracted_2['iteration'], df_extracted_2['episode_reward_mean'], marker='o', linestyle='-', color='r', label='computer 3 Reward Mean')
plt.title('Episode Reward for 8 Ports Boards with 2 Types', fontsize=18)
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Reward', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()



# Plotting loss over iterations
plt.figure(figsize=(10, 5))
plt.plot(df_extracted['iteration'], df_extracted['loss'], marker='o', linestyle='-', color='y', label='Loss')
plt.title('Loss Over Iterations for 8 Ports Rectangular Boards', fontsize=18)
plt.xlabel('Iteration', fontsize=18)
plt.ylabel('Loss', fontsize=18)
plt.legend(fontsize=14)
plt.grid(True)
plt.show()
