
############################################################################################
# The optimal placement are storage in a csv file after running the code 
# to retrive the best options use this code 
## It woks along the board_cecker code 
############################################################################################

import pandas as pd
from collections import Counter
import ast
import matplotlib.pyplot as plt

user_name = r"C:\Users\emago"
csv_file_paths = [
    user_name + r"\Documents\PI_MARL_R\H-board\Qmix\result_Qmix\rect_1decap_dell\Qmix_termination_state_150_R.csv",
    user_name + r"\Documents\PI_MARL_R\H-board\Qmix\result_Qmix\rect_1decp_len\Qmix_termination_state_150_R12.csv",
    user_name + r"\Documents\PI_MARL_R\H-board\Qmix\result_Qmix\rect_1decap_desk\Qmix_termination_state_150_R12.csv"
]

# Extract the states column (assuming it's the second column, index 1)
# Function to process a single file
def process_file(file_path):
    df = pd.read_csv(file_path)
    states = df['state']
    rewards = df['rewards']
    
    states = states.apply(lambda x: tuple(map(int, x.strip('[]').split())))
    rewards = rewards.apply(lambda x: max(map(float, ast.literal_eval(x).values())))
    
    state_counter = Counter(states)
    most_common_state = state_counter.most_common(1)[0]
    
    states_rewards_df = pd.DataFrame({'state': states, 'reward': rewards})
    max_reward = states_rewards_df[states_rewards_df['state'] == most_common_state[0]]['reward'].max()
    
    best_reward_index = rewards.idxmax()
    best_reward_state = states[best_reward_index]
    best_reward_value = rewards[best_reward_index]
    best_reward_count = state_counter[best_reward_state]
    
    return most_common_state, max_reward, best_reward_state, best_reward_value, best_reward_count

# Process all files and collect results
results = [process_file(file_path) for file_path in csv_file_paths]

# Prepare data for plotting
state_labels = [f"Computer {i+1}" for i in range(len(results))] 
state_counts = [result[0][1] for result in results] 
colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold']

# Plotting the results
plt.figure(figsize=(10, 6))
bars = plt.bar(state_labels, state_counts, color=colors)
plt.xlabel('Computers')
plt.ylabel('Number of Appearances')
plt.title('Optimal Placement 12 Ports Rectangular Board')
# Add text labels above the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), ha='center', va='bottom')

plt.show()


# Print the results
for i, result in enumerate(results, 1):
    print(f"File {i}: Most common state {result[0][0]} appears {result[0][1]} times with max reward: {result[1]}")
print(f"The state with the best reward is {results[0][2]} with a reward of {results[0][3]} and appears {results[0][4]} times.")
