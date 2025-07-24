import pandas as pd
from collections import Counter
import ast
user_name = r"C:\Users\emago"
csv_file_path = user_name + r"\Documents\PI_MARL\H-board\Qmix\result_Qmix\iteration158\Qmix_termination_state_150.csv"


# Load the CSV file into a DataFrame
df = pd.read_csv(csv_file_path)

# Extract the states and rewards columns
states = df['state']
rewards = df['rewards']

# Convert states from string to list of integers
states = states.apply(lambda x: tuple(map(int, x.strip('[]').split())))

# Convert rewards from string to dictionary and then get the maximum reward
rewards = rewards.apply(lambda x: max(map(float, ast.literal_eval(x).values())))

# Count the frequency of each state
state_counter = Counter(states)

# Find the three most common states
most_common_states = state_counter.most_common(5)

states_rewards_df = pd.DataFrame({'state': states, 'reward': rewards})

# Get the maximum reward for each of the three most common states
common_states_rewards = []
for state, count in most_common_states:
    max_reward = states_rewards_df[states_rewards_df['state'] == state]['reward'].max()
    common_states_rewards.append((state, count, max_reward))

# Find the state with the best reward
best_reward_index = rewards.idxmax()
best_reward_state = states[best_reward_index]
best_reward_value = rewards[best_reward_index]

print("The three most frequent states and their maximum rewards are:")
for state, count, max_reward in common_states_rewards:
    print(f"State: {state} appears {count} times with max reward: {max_reward}")

print(f"The state with the best reward is {best_reward_state} with a reward of {best_reward_value}.")