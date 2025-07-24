import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# Path to the results file
user_name = r"C:\Users\emago"
results_file = os.path.join(user_name, r"Documents\PI_MARL\H-board\Qmix\result_Qmix\iteration158\result.json")

# Initialize a list to store the data
data = []

# Read the file and parse each JSON object
with open(results_file, 'r') as file:
    for line in file:
        try:
            # Load each JSON object and append to the list
            json_object = json.loads(line)
            data.append(json_object)
        except json.JSONDecodeError:
            print("Error decoding JSON")

# Convert the list of dictionaries to a DataFrame
df = pd.DataFrame(data)

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

# Plotting episode reward mean over iterations
plt.figure(figsize=(10, 5))
plt.plot(df_extracted['iteration'], df_extracted['episode_reward_mean'], marker='o', linestyle='-', color='b', label='Episode Reward Mean')
# plt.plot(df_extracted['iteration'], df_extracted['episode_reward_max'], marker='o', linestyle='-', color='g', label='Episode Reward Max')
# plt.plot(df_extracted['iteration'], df_extracted['episode_reward_min'], marker='o', linestyle='-', color='r', label='Episode Reward Min')
plt.title('Episode Reward Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

# Plotting episode reward mean over iterations
plt.figure(figsize=(10, 5))
# plt.plot(df_extracted['iteration'], df_extracted['episode_reward_mean'], marker='o', linestyle='-', color='b', label='Episode Reward Mean')
plt.plot(df_extracted['iteration'], df_extracted['episode_reward_max'], marker='o', linestyle='-', color='g', label='Episode Reward Max')
plt.plot(df_extracted['iteration'], df_extracted['episode_reward_min'], marker='o', linestyle='-', color='r', label='Episode Reward Min')
plt.title('Episode Reward Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Reward')
plt.legend()
plt.grid(True)
plt.show()

# # Plotting num_env_steps_sampled and num_env_steps_trained over iterations
plt.figure(figsize=(10, 5))
plt.plot(df_extracted['iteration'], df_extracted['num_env_steps_sampled'], marker='o', linestyle='-', color='c', label='Num Env Steps Sampled')
plt.plot(df_extracted['iteration'], df_extracted['num_env_steps_trained'], marker='o', linestyle='-', color='m', label='Num Env Steps Trained')
plt.title('Environment Steps Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Steps')
plt.legend()
plt.grid(True)
plt.show()

# Plotting loss over iterations
plt.figure(figsize=(10, 5))
plt.plot(df_extracted['iteration'], df_extracted['loss'], marker='o', linestyle='-', color='y', label='Loss')
plt.title('Loss Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()
